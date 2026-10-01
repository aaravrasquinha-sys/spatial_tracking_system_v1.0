"""
The frozen wire contract. See SCHEMA.md for the prose spec and
docs/schema_examples/ for golden messages -- this module and both
web clients (web/common/validate.js) are all written against those
same golden files, on purpose, so the three can never quietly drift
apart.

Three message types share one WebSocket:

    tracks   -- ~30 Hz, one per processed M4 frame, always sent even
                when tracks is empty (Section 4.3's "stream stalled"
                vs. "nobody here" distinction).
    event    -- sent when it happens: track_start, track_confirmed,
                track_lost, track_end, calib_suspect.
    health   -- 1 Hz: fps, drops, latency, hardware, client count.

Plus two request/response pairs that are NOT part of the ~30Hz
stream:

    scene    -- one-shot, served over plain HTTP GET /api/scene, not
                the socket. Static per run: frame name, map_id,
                camera pose + intrinsics, camera height, which map
                assets exist (Module 1, once it exists) and their
                URLs, walkable grid (or null), schema version.
    ping/pong -- clock-offset estimation for the VR viewer's
                capture-to-photon latency measurement. The client
                sends its own clock reading; the server echoes it
                back alongside its own. The client keeps the sample
                with the lowest round-trip time.

Phase A (no Module 1/2 yet) still speaks poi.v1. The provisional
frame is signalled entirely by the envelope's own "frame" field
("local" vs "room") and "map_id" (null in Phase A) -- there is no
separate Phase-A wire shape. Both viewers gate on "frame", not on
guessing. Fields Phase A's WorldTrackPhaseA record doesn't have
("conf", "bbox") go out as JSON null, never a fabricated value.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCHEMA_VERSION = "poi.v1"

# Track lifecycle states a wire track can be in. Matches
# poi_localization.tracking.track_manager's _Track.state exactly --
# this module does not invent a fifth state.
TRACK_STATES = ("tentative", "confirmed", "coasting", "lost")

# Where a track's position measurement came from this frame, or
# "predicted" if the Kalman filter coasted with no fresh measurement.
TRACK_SOURCES = ("depth", "raycast", "fused", "predicted")

EVENT_KINDS = ("track_start", "track_confirmed", "track_lost", "track_end", "calib_suspect", "id_switch")

CALIB_STATES = ("ok", "suspect", "missing")
CLOCK_STATES = ("synced", "device")


def _round(x: Optional[float], nd: int) -> Optional[float]:
    return None if x is None else round(float(x), nd)


@dataclass
class TrackWire:
    """One tracks[] entry. Field names match the full system plan's
    Section 4.3 exactly (id, state, p, v, cov_xy, height, conf, src,
    age_s, bbox) regardless of phase -- Phase A just sends null for
    the two fields (conf, bbox) it doesn't have."""

    id: int
    state: str
    p: Tuple[float, float, float]
    v: Tuple[float, float, float]
    cov_xy: Tuple[float, float, float]
    height: Optional[float]
    conf: Optional[float]
    src: str
    age_s: float
    bbox: Optional[Tuple[float, float, float, float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state,
            "p": [_round(self.p[0], 4), _round(self.p[1], 4), _round(self.p[2], 4)],
            "v": [_round(self.v[0], 4), _round(self.v[1], 4), _round(self.v[2], 4)],
            "cov_xy": [_round(c, 6) for c in self.cov_xy],
            "height": _round(self.height, 3),
            "conf": _round(self.conf, 3),
            "src": self.src,
            "age_s": _round(self.age_s, 2),
            "bbox": None if self.bbox is None else [_round(b, 1) for b in self.bbox],
        }


def tracks_message(
    *,
    map_id: Optional[str],
    cam_id: str,
    frame: str,
    seq: int,
    t_capture: float,
    t_publish: float,
    tracks: Sequence[TrackWire],
) -> Dict[str, Any]:
    assert frame in ("local", "room"), f"unknown frame {frame!r}"
    return {
        "type": "tracks",
        "schema": SCHEMA_VERSION,
        "map_id": map_id,
        "cam_id": cam_id,
        "frame": frame,
        "seq": seq,
        "t_capture": t_capture,
        "t_publish": t_publish,
        "tracks": [t.to_dict() for t in tracks],
    }


def event_message(*, event: str, track_id: int, t: float, detail: Optional[str] = None) -> Dict[str, Any]:
    assert event in EVENT_KINDS, f"unknown event kind {event!r}"
    msg: Dict[str, Any] = {"type": "event", "event": event, "id": track_id, "t": t}
    if detail is not None:
        msg["detail"] = detail
    return msg


def health_message(
    *,
    fps: float,
    calib: str,
    map_id: Optional[str],
    temp_c: Optional[float],
    drop_rate: float,
    latency_ms: Optional[float],
    clients: int,
    clock: str = "synced",
) -> Dict[str, Any]:
    assert calib in CALIB_STATES, f"unknown calib state {calib!r}"
    assert clock in CLOCK_STATES, f"unknown clock state {clock!r}"
    return {
        "type": "health",
        "fps": round(fps, 1),
        "calib": calib,
        "map_id": map_id,
        "temp_c": None if temp_c is None else round(temp_c, 1),
        "drop_rate": round(drop_rate, 4),
        "latency_ms": None if latency_ms is None else round(latency_ms, 1),
        "clients": clients,
        "clock": clock,
    }


def pong_message(*, client_t: float, server_t: float) -> Dict[str, Any]:
    return {"type": "pong", "client_t": client_t, "server_t": server_t}


@dataclass
class SceneAsset:
    kind: str  # "mesh" | "points"
    url: str


@dataclass
class SceneDoc:
    """GET /api/scene -- one-shot, not on the socket. Everything a
    client needs once, at connect time, to make sense of the stream:
    which frame it's in, where the camera is, what map assets (if
    any -- Module 1 may not exist yet) it can load, and the walkable
    grid (or null in Phase A)."""

    schema: str
    frame: str
    map_id: Optional[str]
    cam_id: str
    camera_R: List[List[float]]  # 3x3, p_floor = R @ p_cam + t
    camera_t: List[float]  # 3
    camera_height_m: float
    intrinsics: Dict[str, float]  # fx, fy, cx, cy, width, height
    fov_deg: Tuple[float, float]  # (horizontal, vertical), derived from intrinsics
    assets: List[SceneAsset]
    walkable: Optional[Dict[str, Any]]  # {"origin": [x,y], "resolution": m, "grid": [[bool,...],...]} or None
    server_time: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "frame": self.frame,
            "map_id": self.map_id,
            "cam_id": self.cam_id,
            "camera": {
                "R": self.camera_R,
                "t": self.camera_t,
                "height_m": round(self.camera_height_m, 4),
            },
            "intrinsics": self.intrinsics,
            "fov_deg": [round(self.fov_deg[0], 2), round(self.fov_deg[1], 2)],
            "assets": [{"kind": a.kind, "url": a.url} for a in self.assets],
            "walkable": self.walkable,
            "server_time": self.server_time,
        }
