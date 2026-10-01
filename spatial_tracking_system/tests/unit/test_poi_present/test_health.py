import time

from poi_present.config import HealthConfig
from poi_present.server.health import HealthTracker


def test_fps_and_latency_from_frame_stream():
    cfg = HealthConfig(clock_skew_warn_s=2.0)
    tracker = HealthTracker(cfg, window_s=10.0)
    t0 = time.time()
    for i in range(30):
        t_capture = t0 + i / 30.0
        t_publish = t_capture + 0.01  # 10ms pipeline latency
        tracker.record_frame(t_capture, t_publish)
    snap = tracker.snapshot()
    assert 25.0 < snap.fps < 32.0
    assert snap.latency_ms is not None
    assert 5.0 < snap.latency_ms < 20.0
    assert snap.clock == "synced"


def test_clock_skew_detected_and_hides_latency():
    cfg = HealthConfig(clock_skew_warn_s=2.0)
    tracker = HealthTracker(cfg, window_s=10.0)
    # t_capture is "device time" (e.g. seconds since boot), wildly
    # different from wall-clock epoch time.
    tracker.record_frame(t_capture=42.0, t_publish=time.time())
    snap = tracker.snapshot()
    assert snap.clock == "device"
    assert snap.latency_ms is None


def test_drop_rate_reflects_client_drops():
    cfg = HealthConfig()
    tracker = HealthTracker(cfg)
    t0 = time.time()
    for i in range(10):
        tracker.record_frame(t0 + i * 0.033, t0 + i * 0.033 + 0.01)
    tracker.record_client_drops(5)
    snap = tracker.snapshot()
    assert 0.0 < snap.drop_rate < 1.0
