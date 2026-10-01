"""The ONE additive edit to an existing module (poi_present/server/app.py): calib_provider + publish_event_threadsafe."""
import asyncio
import json
import socket
import threading

import pytest
import websockets

from poi_present.config import PresentConfig, ServerConfig
from poi_present.server.app import PresentServer
from poi_present.sources.base import TrackSource
from sts.calib_health import CalibState


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]


class Idle(TrackSource):
    """Emits nothing, so the server's scene.frame is whatever the config says (the synthetic source would
    overwrite it with its own 'local' frame name before the first health tick)."""
    async def run(self, sink):
        await asyncio.sleep(3600)


def _server(port, frame="local"):
    cfg = PresentConfig()
    cfg.server = ServerConfig(host="127.0.0.1", port=port, token=None, max_clients=4)
    cfg.scene.frame = frame
    cfg.health.tick_hz = 20.0
    return PresentServer(cfg, source=Idle())


async def _collect(port, want, n=200):
    got = {}
    async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as ws:
        for _ in range(n):
            m = json.loads(await asyncio.wait_for(ws.recv(), 3.0))
            got.setdefault(m["type"], []).append(m)
            if want(got):
                break
    return got


@pytest.mark.asyncio
async def test_default_behaviour_is_unchanged_without_a_provider():
    for frame, expected in (("local", "missing"), ("room", "ok")):
        port = _port(); srv = _server(port, frame)
        assert srv.calib_provider is None
        task = asyncio.create_task(srv.serve_forever()); await asyncio.sleep(0.1)
        try:
            got = await _collect(port, lambda g: "health" in g)
            assert got["health"][0]["calib"] == expected
        finally:
            srv.stop(); await asyncio.wait_for(task, 5)


@pytest.mark.asyncio
async def test_provider_drives_health_calib_live_and_schema_valid():
    jsonschema = pytest.importorskip("jsonschema")
    from sts.contracts import validate
    port = _port(); srv = _server(port, "room"); cs = CalibState("ok"); srv.calib_provider = cs
    task = asyncio.create_task(srv.serve_forever()); await asyncio.sleep(0.1)
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as ws:
            seen = []
            cs.set("suspect", "moved")
            for _ in range(200):
                m = json.loads(await asyncio.wait_for(ws.recv(), 3.0))
                if m["type"] == "health":
                    assert validate(m, "poi_health") == []
                    seen.append(m["calib"])
                    if m["calib"] == "suspect":
                        break
            assert seen[-1] == "suspect"
    finally:
        srv.stop(); await asyncio.wait_for(task, 5)


@pytest.mark.asyncio
async def test_a_broken_provider_never_reads_as_ok():
    class Broken:
        def state(self): raise RuntimeError("x")
    port = _port(); srv = _server(port, "room"); srv.calib_provider = Broken()
    task = asyncio.create_task(srv.serve_forever()); await asyncio.sleep(0.1)
    try:
        got = await _collect(port, lambda g: "health" in g)
        assert got["health"][0]["calib"] == "suspect"
    finally:
        srv.stop(); await asyncio.wait_for(task, 5)


@pytest.mark.asyncio
async def test_events_can_be_injected_from_another_thread():
    jsonschema = pytest.importorskip("jsonschema")
    from poi_present.schema import event_message
    from sts.contracts import validate
    port = _port(); srv = _server(port); task = asyncio.create_task(srv.serve_forever()); await asyncio.sleep(0.1)
    try:
        assert _port and srv.publish_event_threadsafe({"type": "event"}) in (True, False)
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as ws:
            await asyncio.sleep(0.1)
            msg = event_message(event="calib_suspect", track_id=-1, t=1.0, detail="moved 5 cm")
            ok = {}
            t = threading.Thread(target=lambda: ok.setdefault("v", srv.publish_event_threadsafe(msg))); t.start(); t.join()
            assert ok["v"] is True
            for _ in range(300):
                m = json.loads(await asyncio.wait_for(ws.recv(), 3.0))
                if m["type"] == "event" and m["event"] == "calib_suspect":
                    assert validate(m, "poi_event") == [] and m["detail"] == "moved 5 cm"
                    return
            pytest.fail("calib_suspect event never arrived")
    finally:
        srv.stop(); await asyncio.wait_for(task, 5)


def test_publish_before_the_loop_exists_returns_false():
    assert _server(_port()).publish_event_threadsafe({"type": "event"}) is False
