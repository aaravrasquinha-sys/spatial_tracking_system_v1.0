"""
Read EVERY accel/gyro sample out of a RealSense recording (.bag / .db3).

Why this exists: when a recording is replayed through rs.pipeline() +
wait_for_frames(), the SDK's frame syncer bundles at most ONE accel and ONE
gyro sample into each colour/depth frameset and drops the rest. On room2.db3
(400 Hz gyro, 30 fps video) the pipeline therefore saw about 1 gyro sample
per frame instead of about 13 -- roughly 90% of the IMU data never reached
the mapper, even though the file holds all of it.

This module plays the file back a second time with ONLY the motion sensor
started (sensor-level callback, no syncer), collects every sample, and
returns them as sorted numpy arrays. Timestamps are frame.get_timestamp()
in seconds -- the same clock RealSenseSource uses for colour frames.

    python -m pyslam.sensors.imu_preload ~/bags/room2.db3     # quick check
"""
from __future__ import annotations
import sys
import threading
import time
from typing import Optional, Tuple

import numpy as np

from pyslam.core.log import get_logger

log = get_logger("sensors.imu_preload")


def preload_imu(path: str, timeout_s: float = 900.0
                ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Returns (gyro, accel), each (N,4) float64 [t_s, x, y, z] sorted by t,
    duplicates removed -- or None if the file has no motion sensor or no
    samples could be read (the caller then keeps its old behaviour)."""
    import pyrealsense2 as rs

    ctx = rs.context()
    dev = ctx.load_device(path)
    pb = dev.as_playback()
    pb.set_real_time(False)

    motion_sensor, profiles = None, []
    for s in dev.query_sensors():
        ps = [p for p in s.get_stream_profiles()
              if p.stream_type() in (rs.stream.gyro, rs.stream.accel)]
        if ps:
            motion_sensor, profiles = s, ps
            break
    if motion_sensor is None:
        log.warning(f"{path}: no accel/gyro streams in this recording -- IMU preload skipped.")
        return None

    gyro, accel = [], []
    lock = threading.Lock()

    def _cb(frame) -> None:
        try:
            mf = frame.as_motion_frame()
            d = mf.get_motion_data()
            row = (mf.get_timestamp() / 1000.0, d.x, d.y, d.z)
            is_gyro = mf.get_profile().stream_type() == rs.stream.gyro
        except Exception:  # noqa: BLE001 -- never let one odd frame kill the SDK thread
            return
        with lock:
            (gyro if is_gyro else accel).append(row)

    t0 = time.time()
    motion_sensor.open(profiles)
    motion_sensor.start(_cb)
    try:
        last_n, last_change, last_msg = -1, time.time(), time.time()
        while time.time() - t0 < timeout_s:
            time.sleep(0.2)
            with lock:
                n = len(gyro) + len(accel)
            now = time.time()
            if n != last_n:
                last_n, last_change = n, now
            if now - last_msg > 10.0:
                log.info(f"IMU preload: {n} samples read so far ({now - t0:.0f}s)")
                last_msg = now
            stopped = pb.current_status() == rs.playback_status.stopped
            if stopped and n > 0 and now - last_change > 0.5:
                break
            if now - last_change > 30.0:       # nothing new for 30 s: give up waiting
                log.warning("IMU preload: no new samples for 30 s -- stopping early.")
                break
    finally:
        try:
            motion_sensor.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            motion_sensor.close()
        except Exception:  # noqa: BLE001
            pass

    with lock:
        g = np.array(gyro, dtype=np.float64).reshape(-1, 4)
        a = np.array(accel, dtype=np.float64).reshape(-1, 4)
    if g.shape[0] == 0 or a.shape[0] < 2:
        log.warning(f"{path}: IMU preload read {g.shape[0]} gyro / {a.shape[0]} accel samples -- "
                    f"not usable, keeping the old behaviour.")
        return None
    g = g[np.argsort(g[:, 0], kind="stable")]
    a = a[np.argsort(a[:, 0], kind="stable")]
    g = g[np.concatenate([[True], np.diff(g[:, 0]) > 0])]
    a = a[np.concatenate([[True], np.diff(a[:, 0]) > 0])]
    span = max(g[-1, 0] - g[0, 0], 1e-9)
    log.info(f"IMU preload: {g.shape[0]} gyro ({g.shape[0] / span:.0f} Hz) + {a.shape[0]} accel "
             f"samples over {span:.1f}s, read in {time.time() - t0:.1f}s")
    return g, a


def pair_preloaded(gyro: np.ndarray, accel: np.ndarray, t_prev: float, t_cur: float) -> np.ndarray:
    """(N,7) [t, gx,gy,gz, ax,ay,az] for gyro samples with t_prev < t <= t_cur,
    accel linearly interpolated onto the gyro times -- the same layout and
    window convention as RealSenseSource._pair_imu_since / Frame.imu."""
    i0 = int(np.searchsorted(gyro[:, 0], t_prev, side="right"))
    i1 = int(np.searchsorted(gyro[:, 0], t_cur, side="right"))
    if i1 <= i0:
        return np.zeros((0, 7), dtype=np.float64)
    g = gyro[i0:i1]
    out = np.empty((g.shape[0], 7), dtype=np.float64)
    out[:, :4] = g
    for k in range(3):
        out[:, 4 + k] = np.interp(g[:, 0], accel[:, 0], accel[:, 1 + k])
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m pyslam.sensors.imu_preload FILE.db3")
    res = preload_imu(sys.argv[1])
    if res is None:
        sys.exit("no IMU data could be read")
    g, a = res
    dt = np.diff(g[:, 0])
    print(f"gyro : {g.shape[0]} samples, {g[0, 0]:.3f} .. {g[-1, 0]:.3f} s, "
          f"median gap {np.median(dt) * 1000:.2f} ms, max gap {dt.max() * 1000:.1f} ms")
    print(f"accel: {a.shape[0]} samples")
    still = np.linalg.norm(g[:, 1:4], axis=1)
    print(f"gyro |w| median {np.degrees(np.median(still)):.2f} deg/s, "
          f"first 1 s mean {np.degrees(g[g[:, 0] < g[0, 0] + 1.0, 1:4].mean(0)).round(3)} deg/s")
