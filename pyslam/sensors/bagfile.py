"""
Own bag format: one .npz per short recording. Simple, dependency-free,
and byte-reproducible on replay. Not meant to compete with rosbag; it
exists purely so that hardware only has to be touched once (Phase 0)
and everything after that iterates on recorded data.

Layout inside the npz:
  meta: json string -> {fx,fy,cx,cy,width,height,depth_scale,baseline,n_frames}
  t_<i>: float64 scalar
  rgb_<i>: HxWx3 uint8
  depth_<i>: HxW uint16
  imu_<i>: (N,7) float64  (may be zero rows)
"""
from __future__ import annotations
from typing import Iterator, Optional
import json
import numpy as np

from pyslam.core.types import Frame, Intrinsics, SensorSource
from pyslam.core.log import get_logger

log = get_logger("sensors.bagfile")


class BagWriter:
    def __init__(self, path: str, intr: Intrinsics):
        self.path = path
        self.intr = intr
        self._buf: dict = {}
        self._n = 0

    def write_frame(self, frame: Frame) -> None:
        i = self._n
        self._buf[f"t_{i}"] = np.array([frame.t], dtype=np.float64)
        self._buf[f"rgb_{i}"] = frame.rgb
        self._buf[f"depth_{i}"] = frame.depth
        self._buf[f"imu_{i}"] = frame.imu if frame.imu is not None else np.zeros((0, 7))
        self._n += 1

    def close(self) -> None:
        meta = dict(fx=self.intr.fx, fy=self.intr.fy, cx=self.intr.cx, cy=self.intr.cy,
                    width=self.intr.width, height=self.intr.height,
                    depth_scale=self.intr.depth_scale, baseline=self.intr.baseline,
                    n_frames=self._n)
        self._buf["meta"] = json.dumps(meta)
        np.savez_compressed(self.path, **self._buf)
        log.info(f"Wrote {self._n} frames to {self.path}")


class BagReader:
    """Implements SensorSource by replaying a recorded .npz bag."""

    def __init__(self, path: str):
        self.path = path
        self._data = np.load(path, allow_pickle=False)
        meta = json.loads(str(self._data["meta"]))
        self.n_frames = meta["n_frames"]
        self.intr = Intrinsics(fx=meta["fx"], fy=meta["fy"], cx=meta["cx"], cy=meta["cy"],
                                width=meta["width"], height=meta["height"],
                                depth_scale=meta["depth_scale"], baseline=meta["baseline"])

    def intrinsics(self) -> Intrinsics:
        return self.intr

    def __iter__(self) -> Iterator[Frame]:
        for i in range(self.n_frames):
            t = float(self._data[f"t_{i}"][0])
            rgb = self._data[f"rgb_{i}"]
            depth = self._data[f"depth_{i}"]
            imu = self._data[f"imu_{i}"]
            imu = imu if imu.shape[0] > 0 else None
            yield Frame(t=t, rgb=rgb, depth=depth, intr=self.intr, imu=imu, frame_id=i)

    def close(self) -> None:
        pass


def record(source: SensorSource, path: str, max_frames: Optional[int] = None) -> None:
    writer = BagWriter(path, source.intrinsics())
    n = 0
    for frame in source:
        writer.write_frame(frame)
        n += 1
        if max_frames is not None and n >= max_frames:
            break
    writer.close()
