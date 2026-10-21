"""
Load what a finished run_bag.py run left on disk, in the shape the offline
mapper needs: every keyframe (descriptors, 3D points, odometry pose) from the
run's reloc_map pack, the odometry/bridge links from graph.g2o, the per-frame
odometry trajectory and the per-frame tracking status.
"""
from __future__ import annotations
import json
import os
from typing import Dict, List, Tuple
import numpy as np

from pyslam.core import lie
from pyslam.offline import pgo


def read_tum(path: str) -> Tuple[np.ndarray, List[np.ndarray]]:
    ts, Ts = [], []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            v = [float(x) for x in line.split()[:8]]
            ts.append(v[0])
            q_wxyz = np.array([v[7], v[4], v[5], v[6]])
            q_wxyz /= np.linalg.norm(q_wxyz)
            Ts.append(lie.make_T(lie.quat_to_rot(q_wxyz), np.array(v[1:4])))
    return np.array(ts), Ts


def write_tum(path: str, rows) -> None:
    with open(path, "w") as f:
        f.write("# t tx ty tz qx qy qz qw\n")
        for t, T in sorted(rows, key=lambda r: r[0]):
            q = lie.rot_to_quat(T[:3, :3])               # w x y z
            x = T[:3, 3]
            f.write(f"{t:.9f} {x[0]:.9f} {x[1]:.9f} {x[2]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} {q[0]:.9f}\n")


def read_g2o_edges(path: str) -> List[dict]:
    out = []
    with open(path) as f:
        for line in f:
            if not line.startswith("EDGE_SE3:QUAT"):
                continue
            body, _, comment = line.partition("#")
            v = body.split()
            a, b = int(v[1]), int(v[2])
            x = np.array([float(s) for s in v[3:6]])
            qx, qy, qz, qw = (float(s) for s in v[6:10])
            tri = [float(s) for s in v[10:31]]
            info = np.zeros((6, 6))
            k = 0
            for i in range(6):
                for j in range(i, 6):
                    info[i, j] = info[j, i] = tri[k]
                    k += 1
            kind = comment.strip().replace("kind=", "") or "unknown"
            out.append({"a": a, "b": b, "kind": kind,
                        "T_ab": lie.make_T(lie.quat_to_rot(np.array([qw, qx, qy, qz]) / np.linalg.norm([qw, qx, qy, qz])), x), "info": info})
    return out


def odom_info(T_ab: np.ndarray, sigma_t: float, sigma_t_per_m: float,
              sigma_r_deg: float, sigma_r_per_rad: float) -> np.ndarray:
    """Physically meaningful odometry link weight: a few cm / under a degree per keyframe step,
    growing with the size of the step. (The live pipeline's odometry info is
    eye(6) * inlier_ratio * 100 -- the same number for metres and radians, ~10 cm / ~6 deg --
    which made each loop closure ~100x weaker than one odometry link.)"""
    xi = lie.se3_log(T_ab)
    st = sigma_t + sigma_t_per_m * float(np.linalg.norm(T_ab[:3, 3]))
    sr = np.radians(sigma_r_deg) + sigma_r_per_rad * float(np.linalg.norm(xi[3:]))
    return np.diag([1 / st ** 2] * 3 + [1 / sr ** 2] * 3)


class RunData:
    def __init__(self, run_dir: str):
        from pyslam.mapping.reloc_map import RelocMap
        self.run_dir = run_dir
        pack_dir = os.path.join(run_dir, "reloc_map")
        if not os.path.exists(os.path.join(pack_dir, "meta.json")):
            raise FileNotFoundError(f"{pack_dir} missing -- this run did not finish its export "
                                    f"(the offline mapper needs reloc_map/ from a completed run_bag.py run)")
        self.pack = RelocMap(pack_dir)
        self.intr = self.pack.intrinsics
        self.nodes: Dict[int, object] = {int(i): self.pack.get_node(int(i)) for i in self.pack.node_ids}
        self.ids = sorted(self.nodes)
        self.kf_t = np.array([float(self.nodes[i].sig.t) for i in self.ids])
        self.pose_odom = {i: np.asarray(self.nodes[i].pose_odom, dtype=np.float64) for i in self.ids}
        self.online_final = {int(k): np.asarray(v, dtype=np.float64) for k, v in self.pack.poses.items()}
        self.g2o_edges = read_g2o_edges(os.path.join(run_dir, "graph.g2o"))
        self.frame_t, self.frame_odom = read_tum(os.path.join(run_dir, "trajectory_odom.tum"))
        self.frame_status = {}
        tel_path = os.path.join(run_dir, "telemetry.json")
        if os.path.exists(tel_path):
            with open(tel_path) as f:
                tel = json.load(f)
            for r in tel:
                self.frame_status[round(float(r["t"]), 4)] = r.get("odom_status")

    def base_edges(self, sigma_t=0.01, sigma_t_per_m=0.03, sigma_r_deg=0.3, sigma_r_per_rad=0.03,
                   reloc_info=None) -> Tuple[List[pgo.Edge], dict]:
        """Odometry + bridge (+ local_reloc) links from graph.g2o, re-weighted. The run's own loop /
        proximity links are dropped: the offline search finds its own."""
        keep, counts = [], {}
        idset = set(self.ids)
        for e in self.g2o_edges:
            counts[e["kind"]] = counts.get(e["kind"], 0) + 1
            if e["a"] not in idset or e["b"] not in idset:
                continue
            if e["kind"] == "odom":
                info = odom_info(e["T_ab"], sigma_t, sigma_t_per_m, sigma_r_deg, sigma_r_per_rad)
            elif e["kind"] == "bridge":
                info = e["info"]           # the bridge's own physical sigmas (weak, by design)
            elif e["kind"] == "local_reloc" and reloc_info is not None:
                info = reloc_info
            else:
                continue
            keep.append(pgo.Edge(e["a"], e["b"], e["T_ab"], info, kind=e["kind"]))
        return keep, counts

    def improve_bridges(self, edges: List[pgo.Edge], window_s: float = 1.0, max_speed: float = 1.5,
                        sigma_t: float = 0.05, sigma_t_rel: float = 0.5) -> dict:
        """A bridge spans a lost-tracking gap and the live run guessed 'no movement' across it (or a
        velocity frozen at the moment of loss). Replace its translation by the walking velocity measured
        on the tracked odometry just before AND just after the gap (each in its own keyframe frame,
        so the unknown rotation across the gap does not matter; the bridge's gyro rotation is kept),
        times the gap length. Its weight is set so loop closures can still move it."""
        t_of = {i: float(self.nodes[i].sig.t) for i in self.ids}
        odom_next, odom_prev = {}, {}
        for e in edges:
            if e.kind == "odom":
                odom_next[e.a] = e.b
                odom_prev[e.b] = e.a
        n_fixed, shifts = 0, []
        for e in edges:
            if e.kind != "bridge":
                continue
            dt = t_of[e.b] - t_of[e.a]
            vels = []
            # before: walk back along odometry links from a
            k, Ra = e.a, self.pose_odom[e.a][:3, :3]
            while k in odom_prev and t_of[e.a] - t_of[odom_prev[k]] <= window_s:
                k = odom_prev[k]
            if k != e.a:
                d = Ra.T @ (self.pose_odom[e.a][:3, 3] - self.pose_odom[k][:3, 3])
                vels.append(d / max(t_of[e.a] - t_of[k], 1e-3))
            # after: walk forward from b, expressed in b's frame, rotated into a's frame by the bridge rotation
            k, Rb = e.b, self.pose_odom[e.b][:3, :3]
            while k in odom_next and t_of[odom_next[k]] - t_of[e.b] <= window_s:
                k = odom_next[k]
            if k != e.b:
                d = Rb.T @ (self.pose_odom[k][:3, 3] - self.pose_odom[e.b][:3, 3])
                vels.append(e.T_ab[:3, :3] @ d / max(t_of[k] - t_of[e.b], 1e-3))
            if not vels or dt <= 0:
                continue
            v = np.mean(vels, axis=0)
            if np.linalg.norm(v) > max_speed:
                continue
            t_new = v * dt
            shifts.append(float(np.linalg.norm(t_new - e.T_ab[:3, 3])))
            e.T_ab = e.T_ab.copy()
            e.T_ab[:3, 3] = t_new
            st = sigma_t + sigma_t_rel * float(np.linalg.norm(t_new))
            e.info = e.info.copy()
            e.info[:3, :] = 0.0
            e.info[:, :3] = 0.0
            e.info[:3, :3] = np.eye(3) / st ** 2
            e.meta["velocity_bridge"] = True
            n_fixed += 1
        return {"bridges_with_velocity": n_fixed,
                "bridge_translation_change_m_median": round(float(np.median(shifts)), 3) if shifts else None}

    def frame_poses(self, kf_poses: Dict[int, np.ndarray]) -> List[Tuple[float, np.ndarray]]:
        """Every frame's pose, carried along with its reference keyframe's correction."""
        out = []
        idx = np.searchsorted(self.kf_t, self.frame_t + 1e-6, side="right") - 1
        for t, T_odom, k in zip(self.frame_t, self.frame_odom, idx):
            if k < 0:
                k = 0
            nid = self.ids[k]
            T = kf_poses[nid] @ lie.se3_inverse(self.pose_odom[nid]) @ T_odom
            out.append((float(t), T))
        return out
