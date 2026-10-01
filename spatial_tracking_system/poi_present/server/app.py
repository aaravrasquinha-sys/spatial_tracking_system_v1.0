"""
The publisher. Owns:

  - one asyncio loop, running the configured TrackSource and turning
    each FrameBundle into a `tracks` message, deriving `event`
    messages via adapter.EventDeriver, and broadcasting both;
  - a 1 Hz health tick, broadcasting `health`;
  - HTTP GET for the static dashboard/VR pages, the vendored JS, the
    map bundle (once Module 1 exists), and GET /api/scene;
  - the WebSocket upgrade at /ws, including ping/pong and token auth.

Single port, single dependency (`websockets`, whose asyncio server
answers plain HTTP GETs via `process_request` as well as WS
upgrades) -- see README's "Dependency plan" for why.
"""
from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit, parse_qs

from websockets.asyncio.server import serve, ServerConnection
from websockets.http11 import Response
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed

from poi_present.adapter import EventDeriver, SeqCounter
from poi_present.config import PresentConfig
from poi_present.log import get_logger
from poi_present.schema import CALIB_STATES, SceneDoc, SceneAsset, health_message, pong_message, tracks_message
from poi_present.server.clients import Client, ClientRegistry
from poi_present.server.health import HealthTracker
from poi_present.sources.base import FrameBundle, TrackSource

log = get_logger("server.app")

WS_PATH = "/ws"
SCENE_PATH = "/api/scene"
HEALTHZ_PATH = "/healthz"


@dataclass
class CameraPose:
    R: list  # 3x3
    t: list  # 3
    height_m: float


class SceneState:
    """Mutable because a live run only knows the real camera pose
    once M4's frame_provider has run (Phase A's live calibration
    capture happens at M4 startup, after the server may already be
    listening) -- run_present.py calls update() once that's known.
    Everything else (HTTP serving, the socket) works before that."""

    def __init__(self, cfg: PresentConfig):
        self.cfg = cfg
        sc = cfg.scene
        self.frame = sc.frame
        self.map_id = sc.map_id
        self.cam_id = sc.cam_id
        self.intrinsics = dict(sc.fallback_intrinsics)
        # Identity-ish default: camera at 2.3m looking level. Replaced
        # by the real Phase A/B transform as soon as it's known.
        self.camera = CameraPose(R=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], t=[0.0, 0.0, 2.3], height_m=2.3)
        self.assets: list[SceneAsset] = []
        self.walkable: Optional[dict] = None
        if sc.walkable_grid_path and Path(sc.walkable_grid_path).exists():
            self.walkable = json.loads(Path(sc.walkable_grid_path).read_text())
        if sc.map_bundle_dir:
            self._discover_assets(Path(sc.map_bundle_dir))

    def _discover_assets(self, bundle_dir: Path) -> None:
        glb = bundle_dir / "viewer" / "room.glb"
        pts = bundle_dir / "viewer" / "points.ply"
        if glb.exists():
            self.assets.append(SceneAsset(kind="mesh", url=f"/map/viewer/room.glb"))
        if pts.exists():
            self.assets.append(SceneAsset(kind="points", url=f"/map/viewer/points.ply"))

    def update_camera(self, R, t, height_m: float, frame: str, map_id: Optional[str]) -> None:
        self.camera = CameraPose(R=[list(row) for row in R], t=list(t), height_m=height_m)
        self.frame = frame
        self.map_id = map_id

    def to_doc(self) -> SceneDoc:
        intr = self.intrinsics
        fx, fy = intr["fx"], intr["fy"]
        w, h = intr["width"], intr["height"]
        import math

        fov_h = math.degrees(2 * math.atan(w / (2 * fx)))
        fov_v = math.degrees(2 * math.atan(h / (2 * fy)))
        return SceneDoc(
            schema="poi.v1",
            frame=self.frame,
            map_id=self.map_id,
            cam_id=self.cam_id,
            camera_R=self.camera.R,
            camera_t=self.camera.t,
            camera_height_m=self.camera.height_m,
            intrinsics=intr,
            fov_deg=(fov_h, fov_v),
            assets=self.assets,
            walkable=self.walkable,
            server_time=time.time(),
        )


class PresentServer:
    def __init__(self, cfg: PresentConfig, source: TrackSource):
        self.cfg = cfg
        self.source = source
        self.scene = SceneState(cfg)
        self.registry = ClientRegistry(cfg.server.max_clients, cfg.server.client_queue_size, cfg.server.slow_client_timeout_s)
        self.health = HealthTracker(cfg.health)
        self._events = EventDeriver()
        self._seq = SeqCounter()
        self._static_root = Path(cfg.server.static_dir).resolve()
        self._map_root: Optional[Path] = Path(cfg.scene.map_bundle_dir).resolve() if cfg.scene.map_bundle_dir else None
        self._last_health_json: Optional[str] = None
        self._stop_event: Optional[asyncio.Event] = None
        # STS-merge (additive, default-off): an optional calibration-health provider
        # (.state() -> "ok"|"suspect"|"missing") and the serving loop, so an external
        # supervisor (sts.watchdog_runtime) can drive health.calib and inject events
        # from its own thread. With neither set, behaviour is exactly as before.
        self.calib_provider = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        if cfg.server.token is None and cfg.server.host not in ("127.0.0.1", "localhost", "::1"):
            log.warning(
                "server.token is unset and server.host is not loopback -- anyone on "
                "the LAN can connect. Set server.token before deploying past your own desk "
                "(this stream is a record of movement in your home)."
            )

    # -- ingestion: TrackSource -> broadcast ------------------------------

    async def _on_frame(self, bundle: FrameBundle) -> None:
        t_publish = time.time()
        seq = self._seq.next()
        msg = tracks_message(
            map_id=bundle.map_id,
            cam_id=bundle.cam_id,
            frame=bundle.frame,
            seq=seq,
            t_capture=bundle.t_capture,
            t_publish=t_publish,
            tracks=bundle.tracks,
        )
        self.registry.broadcast_tracks(json.dumps(msg))

        for ev in self._events.diff(bundle.t_capture, bundle.tracks):
            self.registry.broadcast_event(json.dumps(ev))

        self.health.record_frame(bundle.t_capture, t_publish)

        if bundle.frame != self.scene.frame or bundle.map_id != self.scene.map_id:
            self.scene.frame = bundle.frame
            self.scene.map_id = bundle.map_id

    async def _health_loop(self) -> None:
        period = 1.0 / max(self.cfg.health.tick_hz, 0.01)
        while True:
            snap = self.health.snapshot()
            msg = health_message(
                fps=snap.fps,
                calib=self._calib_state(),
                map_id=self.scene.map_id,
                temp_c=snap.temp_c,
                drop_rate=snap.drop_rate,
                latency_ms=snap.latency_ms,
                clients=self.registry.count(),
                clock=snap.clock,
            )
            self._last_health_json = json.dumps(msg)
            self.registry.broadcast_health(self._last_health_json)
            await asyncio.sleep(period)

    def _calib_state(self) -> str:
        # Module 2 / M6's watchdog don't exist in this build yet --
        # Phase A always reports "missing", matching Section 6's
        # discipline ("stream 2D detections flagged, rather than
        # silently publishing wrong 3D positions") once that watchdog
        # is wired in later.
        if self.calib_provider is not None:
            try:
                state = self.calib_provider.state()
            except Exception:  # a broken provider must never read as "ok"
                state = "suspect"
            if state in CALIB_STATES:
                return state
        return "ok" if self.scene.frame == "room" else "missing"

    def publish_event_threadsafe(self, msg: dict) -> bool:
        """Broadcast one already-built `event` message from ANY thread. Returns False if the
        serving loop isn't up yet (the caller should not assume delivery)."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return False
        try:
            loop.call_soon_threadsafe(self.registry.broadcast_event, json.dumps(msg))
            return True
        except RuntimeError:
            return False

    # -- HTTP -------------------------------------------------------------

    def _check_token(self, path_and_query: str) -> bool:
        token = self.cfg.server.token
        if token is None:
            return True
        qs = parse_qs(urlsplit(path_and_query).query)
        supplied = qs.get("token", [None])[0]
        return supplied == token

    def _serve_static(self, path: str) -> Response:
        rel = path.lstrip("/")
        if rel == "" or rel.endswith("/"):
            rel += "index.html"
        candidate = (self._static_root / rel).resolve()
        try:
            candidate.relative_to(self._static_root)
        except ValueError:
            return _text_response(403, "Forbidden")
        if candidate.is_dir():
            candidate = candidate / "index.html"
        if not candidate.is_file():
            return _text_response(404, "Not found")
        body = candidate.read_bytes()
        ctype, _ = mimetypes.guess_type(str(candidate))
        headers = Headers([("Content-Type", ctype or "application/octet-stream"), ("Content-Length", str(len(body)))])
        if "/vendor/" in str(candidate):
            headers["Cache-Control"] = "public, max-age=31536000, immutable"
        else:
            headers["Cache-Control"] = "no-cache"
        return Response(200, "OK", headers, body)

    def _serve_map_asset(self, path: str) -> Response:
        if self._map_root is None:
            return _text_response(404, "No map bundle configured")
        rel = path[len("/map/"):]
        candidate = (self._map_root / rel).resolve()
        try:
            candidate.relative_to(self._map_root)
        except ValueError:
            return _text_response(403, "Forbidden")
        if not candidate.is_file():
            return _text_response(404, "Not found")
        body = candidate.read_bytes()
        ctype, _ = mimetypes.guess_type(str(candidate))
        headers = Headers([
            ("Content-Type", ctype or "application/octet-stream"),
            ("Content-Length", str(len(body))),
            ("Cache-Control", "public, max-age=86400"),
        ])
        return Response(200, "OK", headers, body)

    async def _process_request(self, connection: ServerConnection, request) -> Optional[Response]:
        path_and_query = request.path
        path = urlsplit(path_and_query).path

        if path == WS_PATH:
            if not self._check_token(path_and_query):
                return _text_response(401, "Missing or invalid token")
            return None  # proceed with the WS handshake

        if not self._check_token(path_and_query) and path not in (HEALTHZ_PATH,):
            return _text_response(401, "Missing or invalid token")

        if path == SCENE_PATH:
            body = json.dumps(self.scene.to_doc().to_dict()).encode()
            headers = Headers([("Content-Type", "application/json"), ("Content-Length", str(len(body)))])
            return Response(200, "OK", headers, body)

        if path == HEALTHZ_PATH:
            return _text_response(200, "ok")

        if path.startswith("/map/"):
            return self._serve_map_asset(path)

        return self._serve_static(path)

    # -- WebSocket ----------------------------------------------------------

    async def _ws_handler(self, ws: ServerConnection) -> None:
        client = await self.registry.add(ws)
        if client is None:
            await ws.close(code=1013, reason="server full")
            return
        log.info(f"client {client.client_id} connected ({self.registry.count()}/{self.cfg.server.max_clients})")
        drain_task = asyncio.create_task(client.drain_loop())
        if self._last_health_json is not None:
            client.send_health(self._last_health_json)
        try:
            async for raw in ws:
                await self._handle_client_message(client, raw)
        except ConnectionClosed:
            pass
        finally:
            client.close()
            await drain_task
            await self.registry.remove(client)
            log.info(f"client {client.client_id} disconnected ({self.registry.count()}/{self.cfg.server.max_clients})")

    async def _handle_client_message(self, client: Client, raw) -> None:
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return
        if msg.get("type") == "ping":
            client_t = msg.get("client_t")
            if isinstance(client_t, (int, float)):
                client.send_raw(json.dumps(pong_message(client_t=client_t, server_t=time.time())))

    # -- lifecycle ------------------------------------------------------

    async def serve_forever(self) -> None:
        self._stop_event = asyncio.Event()
        self._loop = asyncio.get_running_loop()
        ssl_context = None
        if self.cfg.server.tls_cert and self.cfg.server.tls_key:
            import ssl

            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(self.cfg.server.tls_cert, self.cfg.server.tls_key)

        async with serve(
            self._ws_handler,
            self.cfg.server.host,
            self.cfg.server.port,
            process_request=self._process_request,
            ssl=ssl_context,
        ) as server:
            scheme = "wss" if ssl_context else "ws"
            log.info(f"poi_present listening on {self.cfg.server.host}:{self.cfg.server.port} "
                      f"({scheme}://.../{WS_PATH.lstrip('/')})")
            source_task = asyncio.create_task(self.source.run(self._on_frame))
            health_task = asyncio.create_task(self._health_loop())
            try:
                await self._stop_event.wait()
            finally:
                self.source.stop()
                source_task.cancel()
                health_task.cancel()
                for t in (source_task, health_task):
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass

    def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()


def _text_response(status: int, text: str) -> Response:
    body = text.encode()
    headers = Headers([("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body)))])
    reason = {200: "OK", 401: "Unauthorized", 403: "Forbidden", 404: "Not Found"}.get(status, "Error")
    return Response(status, reason, headers, body)


def install_signal_handlers(loop: asyncio.AbstractEventLoop, server: PresentServer) -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, server.stop)
        except NotImplementedError:
            pass  # Windows dev machines -- fine to skip, Orin is Linux
