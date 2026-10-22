"""
Fused map from the final per-frame poses.

The live pipeline's map.ply stacks every keyframe's raw depth points on top
of each other, so even a few centimetres of pose error shows up as doubled,
thick walls. Here every used frame's depth is fused into ONE surface:

  * open3d available: a TSDF volume (the standard KinectFusion-style
    fusion RTAB-Map and others export) -> map_mesh.ply (a clean triangle
    mesh) + map_cloud.ply (its points);
  * always also: a voxel-average cloud (each 1-2 cm voxel keeps the mean
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


class _Tsdf:
    """open3d TSDF fusion: the tensor VoxelBlockGrid API (open3d >= 0.15) when present, else the legacy
    ScalableTSDFVolume. Surfaces seen in fewer than min_hits fused frames are left out of the mesh."""
    def __init__(self, o3d, intr, voxel: float, depth_trunc: float):
        self.o3d, self.voxel, self.trunc = o3d, voxel, depth_trunc
        self.scale = 1.0 / intr.depth_scale
        K = np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1]], dtype=np.float64)
        self.tensor = hasattr(o3d, "t") and hasattr(o3d.t.geometry, "VoxelBlockGrid")
        if self.tensor:
            import open3d.core as o3c
            self.o3c = o3c
            self.dev = o3c.Device("CPU:0")
            self.vbg = o3d.t.geometry.VoxelBlockGrid(
                attr_names=("tsdf", "weight", "color"), attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
                attr_channels=((1), (1), (3)), voxel_size=voxel, block_resolution=8,
                block_count=20000, device=self.dev)   # ~200 MB to start; the hash map grows if needed
            self.K = o3c.Tensor(K, o3c.float64)
        else:
            self.vol = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=voxel, sdf_trunc=4.0 * voxel,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
            self.cam = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy)

    def integrate(self, rgb: np.ndarray, depth: np.ndarray, T_world_cam: np.ndarray) -> None:
        o3d = self.o3d
        ext = np.linalg.inv(T_world_cam)
        if self.tensor:
            o3c = self.o3c
            d = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(depth)))
            c = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(rgb)))
            E = o3c.Tensor(ext, o3c.float64)
            blocks = self.vbg.compute_unique_block_coordinates(d, self.K, E, self.scale, self.trunc)
            self.vbg.integrate(blocks, d, c, self.K, self.K, E, self.scale, self.trunc)
        else:
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(rgb)), o3d.geometry.Image(np.ascontiguousarray(depth)),
                depth_scale=self.scale, depth_trunc=self.trunc, convert_rgb_to_intensity=False)
            self.vol.integrate(rgbd, self.cam, ext)

    def write_mesh(self, path: str, min_hits: int) -> int:
        o3d = self.o3d
        if self.tensor:
            mesh = self.vbg.extract_triangle_mesh(weight_threshold=float(min_hits)).to_legacy()
        else:
            mesh = self.vol.extract_triangle_mesh()
        if len(mesh.triangles) == 0:
            return 0
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(path, mesh)
        return len(mesh.triangles)


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
    tsdf = None
    if o3d is not None:
        try:
            tsdf = _Tsdf(o3d, intr, voxel, depth_trunc)
        except Exception as e:  # noqa: BLE001
            log.warning(f"open3d TSDF unavailable ({type(e).__name__}: {e}); voxel cloud only")
    # the voxel-average cloud is always built too: cheap, dependency-free, and the fallback
    # if this machine's open3d cannot produce a mesh
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
            if tsdf is not None:
                try:
                    tsdf.integrate(frame.rgb, frame.depth, T)
                except Exception as e:  # noqa: BLE001
                    log.warning(f"open3d TSDF failed ({type(e).__name__}: {e}); continuing with the voxel cloud only")
                    tsdf = None
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
    if n_used == 0:
        log.error(f"NO frame was fused ({n_seen} read, {n_nopose} without a pose, {n_lost} lost): "
                  f"the recording does not match this run folder (wrong --bag?) or the timestamps differ")
    pts, rgb = fuser.result()
    cloud_path = os.path.join(out_dir, "map_cloud.ply")
    _write_ply_points(cloud_path, pts, rgb)
    out.update({"method": f"voxel average ({voxel / 2 * 100:.0f} cm, >= {min_hits} frames)",
                "cloud": cloud_path, "n_points": int(pts.shape[0])})
    if tsdf is not None:
        try:
            mesh_path = os.path.join(out_dir, "map_mesh.ply")
            n_tri = tsdf.write_mesh(mesh_path, min_hits)
            if n_tri > 0:
                out.update({"method": out["method"] + " + tsdf mesh (open3d)", "mesh": mesh_path,
                            "n_triangles": n_tri})
            else:
                log.warning("open3d produced an empty mesh; use map_cloud.ply")
        except Exception as e:  # noqa: BLE001
            log.warning(f"mesh export failed ({type(e).__name__}: {e}); use map_cloud.ply")
    log.info(f"map: {out}")
    return out
