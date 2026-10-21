"""
Fused map from the final per-frame poses.

The live pipeline's map.ply stacks every keyframe's raw depth points on top
of each other, so even a few centimetres of pose error shows up as doubled,
thick walls. Here every used frame's depth is fused into ONE surface:

  * open3d available: a TSDF volume (the standard KinectFusion-style
    fusion RTAB-Map and others export) -> map_mesh.ply (a clean triangle
    mesh) + map_cloud.ply (its points);
  * otherwise: a voxel-average cloud (each 1-2 cm voxel keeps the mean
    position/colour of everything that fell into it, and voxels seen in
    fewer than min_hits frames are dropped as noise) -> map_cloud.ply.

Frames the tracker marked LOST are skipped: their poses are guesses.
"""
from __future__ import annotations
import os
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from pyslam.core.log import get_logger

log = get_logger("offline.fuse")


def _write_ply_points(path: str, pts: np.ndarray, rgb: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {pts.shape[0]}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        rec = np.empty(pts.shape[0], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                            ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
        rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        f.write(rec.tobytes())


class _VoxelFuser:
    def __init__(self, voxel: float, min_hits: int):
        self.voxel, self.min_hits = voxel, min_hits
        self.keys = np.zeros(0, dtype=np.int64)
        self.acc = np.zeros((0, 7))          # sum x y z r g b, hits
        self._pending: List[Tuple[np.ndarray, np.ndarray]] = []

    def add(self, pts: np.ndarray, rgb: np.ndarray) -> None:
        q = np.floor(pts / self.voxel).astype(np.int64) + (1 << 20)
        k = (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]
        # one vote per voxel per frame, so a voxel's hit count = number of frames that saw it
        k, first = np.unique(k, return_index=True)
        self._pending.append((k, np.hstack([pts[first], rgb[first].astype(np.float64), np.ones((k.size, 1))])))
        if len(self._pending) >= 40:
            self._reduce()

    def _reduce(self) -> None:
        if not self._pending:
            return
        ks = np.concatenate([self.keys] + [k for k, _ in self._pending])
        vs = np.concatenate([self.acc] + [v for _, v in self._pending])
        self._pending = []
        uk, inv = np.unique(ks, return_inverse=True)
        acc = np.zeros((uk.size, 7))
        np.add.at(acc, inv, vs)
        self.keys, self.acc = uk, acc

    def result(self) -> Tuple[np.ndarray, np.ndarray]:
        self._reduce()
        m = self.acc[:, 6] >= self.min_hits
        a = self.acc[m]
        return a[:, :3] / a[:, 6:7], np.clip(a[:, 3:6] / a[:, 6:7], 0, 255).astype(np.uint8)


def fuse_bag(bag_path: str, frame_poses: List[Tuple[float, np.ndarray]], frame_status: Dict[float, str],
             out_dir: str, stride: int = 2, voxel: float = 0.02, depth_trunc: float = 3.0,
             min_hits: int = 3, use_open3d: bool = True, max_frames: Optional[int] = None) -> dict:
    from pyslam.sensors.realsense import RealSenseSource
    ts = np.array([t for t, _ in frame_poses])
    Ts = [T for _, T in frame_poses]

    o3d = None
    if use_open3d:
        try:
            import open3d as o3d  # noqa: F811
        except Exception as e:  # noqa: BLE001
            log.warning(f"open3d not importable ({e}); using the voxel-average cloud instead of a TSDF mesh")
            o3d = None

    src = RealSenseSource(playback_path=bag_path, enable_imu=False, preload_imu=False)
    intr = src.intrinsics()
    vol = fuser = None
    if o3d is not None:
        vol = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel, sdf_trunc=4.0 * voxel,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        cam = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy)
    else:
        fuser = _VoxelFuser(voxel / 2.0, min_hits)
        H, W = intr.height, intr.width
        vv, uu = np.mgrid[0:H:2, 0:W:2]
        xn = (uu - intr.cx) / intr.fx
        yn = (vv - intr.cy) / intr.fy

    n_seen = n_used = n_lost = n_nopose = 0
    t0 = time.time()
    try:
        for frame in src:
            n_seen += 1
            if max_frames is not None and n_seen > max_frames:
                break
            if (n_seen - 1) % stride:
                continue
            k = int(np.searchsorted(ts, frame.t))
            best = None
            for c in (k - 1, k):
                if 0 <= c < ts.size and abs(ts[c] - frame.t) < 1e-3:
                    best = c
            if best is None:
                n_nopose += 1
                continue
            if frame_status.get(round(float(frame.t), 4)) == "LOST":
                n_lost += 1
                continue
            T = Ts[best]
            if o3d is not None:
                depth = frame.depth.copy()
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(np.ascontiguousarray(frame.rgb)), o3d.geometry.Image(depth),
                    depth_scale=1.0 / intr.depth_scale, depth_trunc=depth_trunc,
                    convert_rgb_to_intensity=False)
                vol.integrate(rgbd, cam, np.linalg.inv(T))
            else:
                z = frame.depth[0::2, 0::2].astype(np.float64) * intr.depth_scale
                m = (z > 0.2) & (z < depth_trunc)
                p = np.stack([xn[m] * z[m], yn[m] * z[m], z[m]], axis=1)
                p = p @ T[:3, :3].T + T[:3, 3]
                fuser.add(p, frame.rgb[0::2, 0::2][m])
            n_used += 1
            if n_used % 200 == 0:
                log.info(f"fused {n_used} frames ({n_seen} read, {time.time() - t0:.0f}s)")
    finally:
        try:
            src.close()
        except Exception:  # noqa: BLE001
            pass

    out = {"frames_read": n_seen, "frames_fused": n_used, "frames_skipped_lost": n_lost,
           "frames_without_pose": n_nopose, "voxel_m": voxel, "seconds": round(time.time() - t0, 1)}
    if o3d is not None:
        mesh = vol.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        mesh_path = os.path.join(out_dir, "map_mesh.ply")
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        pcd = vol.extract_point_cloud()
        cloud_path = os.path.join(out_dir, "map_cloud.ply")
        o3d.io.write_point_cloud(cloud_path, pcd)
        out.update({"method": "tsdf (open3d)", "mesh": mesh_path, "cloud": cloud_path,
                    "n_triangles": len(mesh.triangles), "n_points": len(pcd.points)})
    else:
        pts, rgb = fuser.result()
        cloud_path = os.path.join(out_dir, "map_cloud.ply")
        _write_ply_points(cloud_path, pts, rgb)
        out.update({"method": f"voxel average ({voxel / 2 * 100:.0f} cm, >= {min_hits} frames)",
                    "cloud": cloud_path, "n_points": int(pts.shape[0])})
    log.info(f"map: {out}")
    return out
