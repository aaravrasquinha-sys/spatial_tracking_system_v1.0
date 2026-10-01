"""
One client = one bounded outgoing queue + one drain task. Matches
the DropStaleQueue philosophy the rest of this system already uses
(poi_perception/runtime/queues.py): a stale `tracks` message is
worse than no message, so a full queue drops the OLDEST tracks
message to make room for the newest one. `event` messages are never
dropped -- a lost track_start/track_end would desync a client's
track list from reality in a way a stale position never would.

A client that stays backed up (can't keep its queue below its
slow_client_timeout_s deadline) gets disconnected outright, rather
than let one slow phone hold back the whole broadcast.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional
from collections import deque

from poi_present.log import get_logger

log = get_logger("server.clients")


@dataclass
class ClientStats:
    connected_at: float
    messages_sent: int = 0
    tracks_dropped: int = 0
    last_send_t: float = field(default_factory=time.time)


class Client:
    def __init__(self, ws, client_id: int, queue_size: int, slow_timeout_s: float):
        self.ws = ws
        self.client_id = client_id
        self.slow_timeout_s = slow_timeout_s
        self.stats = ClientStats(connected_at=time.time())
        # Two lanes: events (never dropped, small and rare) and a
        # single-slot "latest tracks" lane (drop-oldest). This is the
        # per-client analogue of DropStaleQueue, and it's why a
        # `tracks` message never queues up more than one deep.
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._latest_tracks: Optional[str] = None
        self._latest_health: Optional[str] = None
        self._wake = asyncio.Event()
        self._closed = False

    def send_tracks(self, payload: str) -> None:
        if self._latest_tracks is not None:
            self.stats.tracks_dropped += 1
        self._latest_tracks = payload
        self._wake.set()

    def send_health(self, payload: str) -> None:
        self._latest_health = payload
        self._wake.set()

    def send_event(self, payload: str) -> None:
        self._event_queue.put_nowait(payload)
        self._wake.set()

    def send_raw(self, payload: str) -> None:
        """For one-off replies (pong) -- routed through the event lane
        since it's rare and must not be dropped."""
        self.send_event(payload)

    def close(self) -> None:
        self._closed = True
        self._wake.set()

    async def drain_loop(self) -> None:
        """Owns the actual ws.send() calls for this client. Runs until
        close() or the connection dies. A send that blocks past
        slow_client_timeout_s disconnects the client -- a backed-up
        TCP write buffer (dead Wi-Fi, a backgrounded phone) must not
        hold up this client's own queue forever, let alone anyone
        else's (every client has its own task, so one slow client
        never blocks another)."""
        try:
            while not self._closed:
                await self._wake.wait()
                self._wake.clear()
                while True:
                    sent_anything = False
                    if not self._event_queue.empty():
                        payload = self._event_queue.get_nowait()
                        await self._send_with_timeout(payload)
                        sent_anything = True
                    if self._latest_health is not None:
                        payload, self._latest_health = self._latest_health, None
                        await self._send_with_timeout(payload)
                        sent_anything = True
                    if self._latest_tracks is not None:
                        payload, self._latest_tracks = self._latest_tracks, None
                        await self._send_with_timeout(payload)
                        sent_anything = True
                    if not sent_anything:
                        break
        except _SlowClientTimeout:
            log.warning(f"client {self.client_id}: too slow to keep up, disconnecting")
        except Exception:
            pass  # normal disconnects (ConnectionClosed etc.) end the loop here
        finally:
            try:
                await self.ws.close()
            except Exception:
                pass

    async def _send_with_timeout(self, payload: str) -> None:
        try:
            await asyncio.wait_for(self.ws.send(payload), timeout=self.slow_timeout_s)
            self.stats.messages_sent += 1
            self.stats.last_send_t = time.time()
        except asyncio.TimeoutError:
            raise _SlowClientTimeout()


class _SlowClientTimeout(Exception):
    pass


class ClientRegistry:
    def __init__(self, max_clients: int, queue_size: int, slow_timeout_s: float):
        self.max_clients = max_clients
        self.queue_size = queue_size
        self.slow_timeout_s = slow_timeout_s
        self._clients: Dict[int, Client] = {}
        self._next_id = 1
        self._lock = asyncio.Lock()

    async def add(self, ws) -> Optional[Client]:
        async with self._lock:
            if len(self._clients) >= self.max_clients:
                return None
            cid = self._next_id
            self._next_id += 1
            client = Client(ws, cid, self.queue_size, self.slow_timeout_s)
            self._clients[cid] = client
            return client

    async def remove(self, client: Client) -> None:
        async with self._lock:
            self._clients.pop(client.client_id, None)

    def count(self) -> int:
        return len(self._clients)

    def broadcast_tracks(self, payload: str) -> None:
        for c in list(self._clients.values()):
            c.send_tracks(payload)

    def broadcast_event(self, payload: str) -> None:
        for c in list(self._clients.values()):
            c.send_event(payload)

    def broadcast_health(self, payload: str) -> None:
        for c in list(self._clients.values()):
            c.send_health(payload)
