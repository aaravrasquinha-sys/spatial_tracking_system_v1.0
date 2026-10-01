import asyncio
import json
import socket
import urllib.request

import pytest
import websockets

from poi_present.config import PresentConfig, ServerConfig, SourceConfig
from poi_present.server.app import PresentServer
from poi_present.sources.synthetic import build_synthetic_source


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_config(port: int, token=None) -> PresentConfig:
    cfg = PresentConfig()
    cfg.server = ServerConfig(host="127.0.0.1", port=port, token=token, max_clients=4, client_queue_size=4)
    cfg.source = SourceConfig(kind="synthetic", synthetic_mode="scripted", synthetic_n_walkers=2, synthetic_seed=3)
    return cfg


@pytest.mark.asyncio
async def test_client_receives_tracks_and_health():
    port = _free_port()
    cfg = _make_config(port)
    server = PresentServer(cfg, source=build_synthetic_source(cfg.source, tick_hz=30.0))
    server_task = asyncio.create_task(server.serve_forever())
    await asyncio.sleep(0.1)

    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as ws:
            got_tracks, got_health = False, False
            for _ in range(60):
                raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                msg = json.loads(raw)
                assert msg["type"] in ("tracks", "health", "event")
                if msg["type"] == "tracks":
                    got_tracks = True
                    assert msg["schema"] == "poi.v1"
                    assert msg["frame"] == "local"
                if msg["type"] == "health":
                    got_health = True
                if got_tracks and got_health:
                    break
            assert got_tracks and got_health
    finally:
        server.stop()
        await asyncio.wait_for(server_task, timeout=5.0)


@pytest.mark.asyncio
async def test_scene_endpoint_over_plain_http():
    port = _free_port()
    cfg = _make_config(port)
    server = PresentServer(cfg, source=build_synthetic_source(cfg.source, tick_hz=30.0))
    server_task = asyncio.create_task(server.serve_forever())
    await asyncio.sleep(0.1)

    try:
        loop = asyncio.get_running_loop()

        def _fetch():
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/scene", timeout=2.0) as r:
                return json.loads(r.read())

        doc = await loop.run_in_executor(None, _fetch)
        assert doc["schema"] == "poi.v1"
        assert doc["frame"] == "local"
        assert "intrinsics" in doc
    finally:
        server.stop()
        await asyncio.wait_for(server_task, timeout=5.0)


@pytest.mark.asyncio
async def test_token_required_when_set():
    port = _free_port()
    cfg = _make_config(port, token="s3cret")
    server = PresentServer(cfg, source=build_synthetic_source(cfg.source, tick_hz=30.0))
    server_task = asyncio.create_task(server.serve_forever())
    await asyncio.sleep(0.1)

    try:
        with pytest.raises(Exception):
            async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as ws:
                await ws.recv()

        async with websockets.connect(f"ws://127.0.0.1:{port}/ws?token=s3cret") as ws:
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            assert json.loads(raw)["type"] in ("tracks", "health")
    finally:
        server.stop()
        await asyncio.wait_for(server_task, timeout=5.0)


@pytest.mark.asyncio
async def test_max_clients_enforced():
    port = _free_port()
    cfg = _make_config(port)
    cfg.server.max_clients = 1
    server = PresentServer(cfg, source=build_synthetic_source(cfg.source, tick_hz=30.0))
    server_task = asyncio.create_task(server.serve_forever())
    await asyncio.sleep(0.1)

    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as ws1:
            await asyncio.wait_for(ws1.recv(), timeout=2.0)
            async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as ws2:
                with pytest.raises(Exception):
                    await asyncio.wait_for(ws2.recv(), timeout=2.0)
    finally:
        server.stop()
        await asyncio.wait_for(server_task, timeout=5.0)
