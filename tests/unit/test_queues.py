import threading
import time

from poi_perception.runtime.queues import DropStaleQueue


def test_put_get_single_item():
    q: DropStaleQueue[int] = DropStaleQueue()
    q.put(1)
    assert q.get(timeout=1.0) == 1


def test_put_overwrites_unread_item_drop_stale():
    q: DropStaleQueue[int] = DropStaleQueue()
    q.put(1)
    q.put(2)  # 1 was never read -- must be dropped, not queued
    assert q.get(timeout=1.0) == 2
    assert q.dropped_count == 1


def test_get_blocks_until_put():
    q: DropStaleQueue[int] = DropStaleQueue()
    result = {}

    def producer():
        time.sleep(0.1)
        q.put(42)

    t = threading.Thread(target=producer)
    t.start()
    result["value"] = q.get(timeout=2.0)
    t.join()
    assert result["value"] == 42


def test_get_timeout_returns_none():
    q: DropStaleQueue[int] = DropStaleQueue()
    assert q.get(timeout=0.05) is None


def test_close_unblocks_waiting_get():
    q: DropStaleQueue[int] = DropStaleQueue()
    result = {}

    def closer():
        time.sleep(0.1)
        q.close()

    t = threading.Thread(target=closer)
    t.start()
    result["value"] = q.get(timeout=2.0)
    t.join()
    assert result["value"] is None
    assert q.closed
