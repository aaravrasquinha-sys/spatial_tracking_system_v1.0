import asyncio

import pytest

from poi_present.server.clients import Client, ClientRegistry


class _FakeWS:
    """A minimal stand-in for a websockets ServerConnection: records
    every payload actually sent, and can be told to stall sends for a
    slow-client test."""

    def __init__(self, send_delay: float = 0.0):
        self.sent = []
        self.send_delay = send_delay
        self.closed = False

    async def send(self, payload: str) -> None:
        if self.send_delay:
            await asyncio.sleep(self.send_delay)
        self.sent.append(payload)

    async def close(self, code=None, reason=None) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_tracks_lane_drops_oldest_not_newest():
    ws = _FakeWS(send_delay=0.05)  # slow enough that sends pile up
    client = Client(ws, client_id=1, queue_size=4, slow_timeout_s=5.0)
    drain = asyncio.create_task(client.drain_loop())

    client.send_tracks("frame-1")
    await asyncio.sleep(0.001)  # let drain_loop pick up frame-1 and start sending it
    client.send_tracks("frame-2")
    client.send_tracks("frame-3")  # frame-2 should be dropped in favour of this

    await asyncio.sleep(0.2)
    client.close()
    await drain

    assert "frame-1" in ws.sent
    assert "frame-3" in ws.sent
    assert client.stats.tracks_dropped >= 1


@pytest.mark.asyncio
async def test_events_are_never_dropped():
    ws = _FakeWS()
    client = Client(ws, client_id=1, queue_size=4, slow_timeout_s=5.0)
    drain = asyncio.create_task(client.drain_loop())

    for i in range(10):
        client.send_event(f"event-{i}")

    await asyncio.sleep(0.1)
    client.close()
    await drain

    for i in range(10):
        assert f"event-{i}" in ws.sent


@pytest.mark.asyncio
async def test_slow_client_gets_disconnected():
    ws = _FakeWS(send_delay=0.5)
    client = Client(ws, client_id=1, queue_size=4, slow_timeout_s=0.05)
    drain = asyncio.create_task(client.drain_loop())

    client.send_event("hello")
    await asyncio.wait_for(drain, timeout=2.0)
    assert ws.closed


@pytest.mark.asyncio
async def test_registry_enforces_max_clients():
    registry = ClientRegistry(max_clients=1, queue_size=4, slow_timeout_s=5.0)
    c1 = await registry.add(_FakeWS())
    assert c1 is not None
    c2 = await registry.add(_FakeWS())
    assert c2 is None
    await registry.remove(c1)
    c3 = await registry.add(_FakeWS())
    assert c3 is not None
