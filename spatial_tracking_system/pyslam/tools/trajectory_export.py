"""
WP-T2: trajectory + keyframe-pose exporters. One shared code path for
run_slam.py, run_bag.py and run_synth.py so hardware and synthetic runs
produce the exact same files, and so a synthetic gate (an exported TUM
file must reproduce the ATE computed in-process, see
tests/gates/test_g_traj.py) actually exercises what ships.

Files written into a run directory:
  trajectory_odom.tum        -- online, per-FRAME, raw odometry (no
                                 loop-closure correction at all)
  trajectory_final_frames.tum-- FINAL (post pipeline.finalize()), per-FRAME,
                                 every frame's pose reconstructed as
                                 final_pose(ref_kf) @ T_ref_frame
  trajectory_final_kf.tum    -- FINAL, per-KEYFRAME only (same poses as
                                 above, filtered to is_keyframe rows --
                                 provided separately since most external
                                 tools/eval scripts expect keyframe-rate
                                 trajectories, not the denser per-frame one)
  keyframes.csv / keyframes.json
                              -- see export_keyframes' docstring
  graph.g2o                  -- the optimised pose graph, reopenable
                                 independently of this codebase
  trajectory_report.json     -- path length, start/end gap, height range,
                                 max correction, loop count, gravity-
                                 alignment status

TUM format: "t tx ty tz qx qy qz qw" (space-separated, one pose per
line). NOTE the quaternion order: TUM puts w LAST. This codebase's own
frozen convention (pyslam.core.lie) is [w,x,y,z] -- every writer in this
module does the reorder explicitly at the point of writing, and
test_g_traj.py round-trips a written file back through a TUM reader to
catch a silent regression here, since a w-first/w-last mixup is exactly
the kind of bug that looks fine until read by an external tool.
"""
from __future__ import annotations
from typing import Optional
import json
import os
import numpy as np

from pyslam.core import lie
from pyslam.core.gravity_frame import estimate_gravity_alignment, pose_to_grav


def _quat_wxyz_to_tum_xyzw(q_wxyz: np.ndarray) -> tuple:
    w, x, y, z = q_wxyz
    return (x, y, z, w)


def write_tum(path: str, rows: list[tuple]) -> None:
    """rows: list of (t, T_4x4). Writes 't tx ty tz qx qy qz qw' per line,
    sorted by t (TUM tools generally assume monotonic timestamps)."""
    rows = sorted(rows, key=lambda r: r[0])
    with open(path, "w") as f:
        f.write("# t tx ty tz qx qy qz qw\n")
        for t, T in rows:
            xyz = T[:3, 3]
            q_wxyz = lie.rot_to_quat(T[:3, :3])
            qx, qy, qz, qw = _quat_wxyz_to_tum_xyzw(q_wxyz)
            f.write(f"{t:.9f} {xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f} "
                    f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")


def read_tum(path: str) -> list[tuple]:
    """Inverse of write_tum -- used by test_g_traj.py's round-trip gate,
    and generally useful for anyone auditing an exported file."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [float(x) for x in line.split()]
            t, tx, ty, tz, qx, qy, qz, qw = parts
            R = lie.quat_to_rot(np.array([qw, qx, qy, qz]))
            T = lie.make_T(R, np.array([tx, ty, tz]))
            out.append((t, T))
    return out


def write_g2o(path: str, node_poses: dict, links: list) -> None:
    """Minimal SE3 g2o dump: VERTEX_SE3:QUAT + EDGE_SE3:QUAT, both using
    g2o's own [x y z qx qy qz qw] pose convention (w last, same reorder
    concern as TUM above) and a diagonal-only information block (g2o's
    upper-triangular 21-value layout does support the full matrix, but a
    diagonal approximation is what every commonly-available g2o viewer
    assumes when eyeballing a graph, and the full 6x6 is already
    available in this project's own JSON/pickle outputs for anyone who
    needs it exactly)."""
    with open(path, "w") as f:
        for nid, T in sorted(node_poses.items()):
            xyz = T[:3, 3]
            q_wxyz = lie.rot_to_quat(T[:3, :3])
            qx, qy, qz, qw = _quat_wxyz_to_tum_xyzw(q_wxyz)
            f.write(f"VERTEX_SE3:QUAT {nid} {xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f} "
                    f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")
        for link in links:
            if link.a not in node_poses or link.b not in node_poses:
                continue
            xyz = link.T_ab[:3, 3]
            q_wxyz = lie.rot_to_quat(link.T_ab[:3, :3])
            qx, qy, qz, qw = _quat_wxyz_to_tum_xyzw(q_wxyz)
            diag = np.clip(np.diag(link.info), 1e-9, None)
            # g2o's EDGE_SE3:QUAT information block is the 21 values of
            # the upper triangle, row-major (I11 I12 ... I16 I22 I23 ...
            # I66). A diagonal-only approximation: on-diagonal entries
            # are this link's own info diagonal, every off-diagonal 0.
            upper_tri = []
            for i in range(6):
                for j in range(i, 6):
                    upper_tri.append(diag[i] if i == j else 0.0)
            info_str = " ".join(f"{v:.6f}" for v in upper_tri)
            f.write(f"EDGE_SE3:QUAT {link.a} {link.b} {xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f} "
                    f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f} {info_str}  # kind={link.kind}\n")


def build_frame_final_poses(frame_records: list, final_poses: dict) -> list:
    """WP-T1's reconstruction rule: a frame's FINAL pose is
    final_pose(ref_kf_id) @ T_ref_frame. Frames whose ref_kf_id never
    made it into final_poses (shouldn't happen in a healthy run -- every
    keyframe is added to the graph unconditionally -- but a partial/
    interrupted run could leave one dangling) are skipped, not crashed
    on, with the count reported back so the caller can log it."""
    out = []
    n_skipped = 0
    for rec in frame_records:
        ref = rec["ref_kf_id"]
        if ref is None or ref not in final_poses:
            n_skipped += 1
            continue
        T_final_frame = final_poses[ref] @ rec["T_ref_frame"]
        out.append((rec["t"], rec["frame_id"], T_final_frame, rec["is_keyframe"],
                     rec["status"], rec["session_id"]))
    return out, n_skipped


def export_keyframes(path_csv: str, path_json: str, node_ids: list, memory, graph,
                      final_poses: dict, T_grav_cam0: np.ndarray,
                      loop_events: list, proximity_events: list, status_log: list) -> None:
    """WP-T2's keyframe-by-keyframe file, per the spec agreed in
    conversation: identity, pose in both W_cam0 and W_grav, path,
    quality (odometry inliers, correction magnitude vs raw odometry),
    graph (loop/proximity partners), flags (post-LOST, was-in-LTM)."""
    loop_partners: dict = {}
    for a, b, link in loop_events:
        loop_partners.setdefault(a, []).append(b)
        loop_partners.setdefault(b, []).append(a)
    prox_partners: dict = {}
    for a, b, link in proximity_events:
        prox_partners.setdefault(a, []).append(b)
        prox_partners.setdefault(b, []).append(a)
    after_lost_ids = set()
    for line in status_log:
        # pipeline.py's own bridge-link status_log line names both ids
        # explicitly; a plain LOST line doesn't name a node, so we can
        # only mark the bridge cases here -- see caller for the fuller
        # LOST-adjacent accounting via session_id instead.
        if "bridge link node" in line:
            try:
                b_id = int(line.split("<->")[1].strip().split()[1])
                after_lost_ids.add(b_id)
            except (IndexError, ValueError):
                pass

    odom_by_id = {nid: memory.get(nid).pose_odom for nid in node_ids}
    weight_by_id = {nid: memory.get(nid).weight for nid in node_ids}
    session_by_id = {nid: memory.get(nid).session_id for nid in node_ids}
    t_by_id = {nid: memory.get(nid).sig.t for nid in node_ids}
    ltm_ids = set(memory.ltm_ids_expanded())

    cumulative = 0.0
    prev_xyz = None
    rows = []
    for kf_index, nid in enumerate(sorted(node_ids)):
        T_final = final_poses.get(nid)
        if T_final is None:
            continue
        T_odom = odom_by_id[nid]
        xyz = T_final[:3, 3]
        q_wxyz = lie.rot_to_quat(T_final[:3, :3])

        T_grav = pose_to_grav(T_final, T_grav_cam0)
        xyz_g = T_grav[:3, 3]
        R_g = T_grav[:3, :3]
        # standard ZYX (yaw-pitch-roll) extraction for a human-readable
        # roll/pitch/yaw triple in the gravity-aligned frame; the
        # quaternion above (in W_cam0) remains the lossless pose.
        pitch = np.degrees(np.arcsin(np.clip(-R_g[2, 0], -1.0, 1.0)))
        roll = np.degrees(np.arctan2(R_g[2, 1], R_g[2, 2]))
        yaw = np.degrees(np.arctan2(R_g[1, 0], R_g[0, 0]))

        if prev_xyz is not None:
            cumulative += float(np.linalg.norm(xyz - prev_xyz))
        prev_xyz = xyz

        xi_corr = lie.se3_log(lie.se3_inverse(T_odom) @ T_final)
        correction_cm = float(np.linalg.norm(xi_corr[:3])) * 100.0
        correction_deg = float(np.degrees(np.linalg.norm(xi_corr[3:])))

        row = {
            "kf_index": kf_index, "node_id": nid,
            "timestamp_s": float(t_by_id[nid]),
            "session_id": session_by_id[nid],
            "x_cam0": float(xyz[0]), "y_cam0": float(xyz[1]), "z_cam0": float(xyz[2]),
            "qw": float(q_wxyz[0]), "qx": float(q_wxyz[1]), "qy": float(q_wxyz[2]), "qz": float(q_wxyz[3]),
            "x_grav": float(xyz_g[0]), "y_grav": float(xyz_g[1]), "z_grav": float(xyz_g[2]),
            "roll_deg": float(roll), "pitch_deg": float(pitch), "yaw_deg": float(yaw),
            "cumulative_distance_m": float(cumulative),
            "odom_weight": int(weight_by_id[nid]),
            "correction_cm": correction_cm, "correction_deg": correction_deg,
            "loop_closure_partner_ids": loop_partners.get(nid, []),
            "proximity_partner_ids": prox_partners.get(nid, []),
            "is_after_lost": nid in after_lost_ids,
            "was_in_ltm": nid in ltm_ids,
        }
        rows.append(row)

    with open(path_json, "w") as f:
        json.dump(rows, f, indent=2, default=float)

    if rows:
        cols = list(rows[0].keys())
        with open(path_csv, "w") as f:
            f.write(",".join(cols) + "\n")
            for row in rows:
                vals = []
                for c in cols:
                    v = row[c]
                    if isinstance(v, list):
                        v = "|".join(str(x) for x in v)
                    vals.append(str(v))
                f.write(",".join(vals) + "\n")
    else:
        with open(path_csv, "w") as f:
            f.write("")


def build_trajectory_report(node_ids: list, final_poses: dict, T_grav_cam0: np.ndarray,
                             gravity_aligned: bool, gravity_reason: str,
                             loop_events: list, proximity_events: list,
                             cross_session_merges: list, status_log: list,
                             n_frames_skipped_in_reconstruction: int = 0) -> dict:
    if not node_ids or not final_poses:
        return {"n_keyframes": 0, "note": "no keyframes to report"}
    pts_grav = np.array([pose_to_grav(final_poses[nid], T_grav_cam0)[:3, 3]
                          for nid in sorted(node_ids) if nid in final_poses])
    path_len = float(np.sum(np.linalg.norm(np.diff(pts_grav, axis=0), axis=1))) if len(pts_grav) > 1 else 0.0
    start_end_gap = float(np.linalg.norm(pts_grav[-1] - pts_grav[0])) if len(pts_grav) > 1 else 0.0
    height_min = float(pts_grav[:, 2].min())
    height_max = float(pts_grav[:, 2].max())
    return {
        "n_keyframes": len(pts_grav),
        "path_length_m": path_len,
        "start_to_end_gap_m": start_end_gap,
        "height_range_m": [height_min, height_max],
        # WP-K1: height_range_m is the Z extent in W_grav. When gravity
        # alignment failed (gravity_aligned=False), T_grav_cam0 is the
        # identity and "Z" is simply the FIRST CAMERA's optical axis --
        # i.e. how far the camera travelled straight ahead of where it
        # started, which for a corridor run is mostly ordinary horizontal
        # travel, NOT height. Reading that number as vertical drift is
        # exactly the mistake this flag exists to prevent.
        "height_range_valid": bool(gravity_aligned),
        "height_range_note": ("Z extent in the gravity-aligned frame." if gravity_aligned else
                               "NOT HEIGHT: gravity alignment failed, so this is the extent along the first "
                               "camera's optical axis (ordinary forward travel included). Do not read as drift."),
        "n_loop_closures": len(loop_events),
        "n_proximity_links": len(proximity_events),
        "n_cross_session_merges": len(cross_session_merges),
        "n_lost_events": sum(1 for l in status_log if "LOST" in l and "bridge" not in l),
        "n_bridge_links": sum(1 for l in status_log if "bridge link" in l),
        "gravity_aligned": gravity_aligned,
        "gravity_alignment_reason": gravity_reason,
        "n_frames_skipped_in_final_reconstruction": n_frames_skipped_in_reconstruction,
    }


def export_all(run_dir: str, pipeline, result, source_intr=None) -> tuple:
    """Single entry point called by run_slam.py / run_bag.py / run_synth.py
    right after pipeline.finalize(result). Writes every file listed in
    this module's docstring. Returns (trajectory_report dict, T_grav_cam0)
    -- the report is also written to disk, and T_grav_cam0 is handed
    straight to plots.export_plots so gravity alignment (which needs the
    same startup-IMU-window logic) is only computed once per run."""
    final_poses = result.final_poses or {}
    node_ids = sorted(pipeline.memory.all_node_ids())

    # WP-T3: gravity alignment. R_body_cam comes from pipeline._R_body_cam
    # (identity unless the caller passed a real one -- see run_slam.py).
    align = estimate_gravity_alignment(result.startup_imu_window, pipeline._R_body_cam,
                                        gyro_thresh_rad_s=pipeline.cfg.gravity_prior_gyro_thresh_rad_s,
                                        accel_std_thresh_mps2=pipeline.cfg.gravity_prior_accel_std_thresh_mps2)
    T_grav_cam0 = align.T_grav_cam0

    # -- raw odometry, per frame (no corrections at all) --
    write_tum(os.path.join(run_dir, "trajectory_odom.tum"), result.odom_trajectory)

    # -- final, per-frame and per-keyframe --
    frame_final, n_skipped = build_frame_final_poses(result.frame_records, final_poses)
    all_rows = [(t, T) for (t, fid, T, is_kf, status, sess) in frame_final]
    kf_rows = [(t, T) for (t, fid, T, is_kf, status, sess) in frame_final if is_kf]
    write_tum(os.path.join(run_dir, "trajectory_final_frames.tum"), all_rows)
    write_tum(os.path.join(run_dir, "trajectory_final_kf.tum"), kf_rows)

    # -- g2o dump of the closed graph --
    write_g2o(os.path.join(run_dir, "graph.g2o"), final_poses, pipeline.graph.links)

    # -- keyframes.csv / keyframes.json --
    export_keyframes(os.path.join(run_dir, "keyframes.csv"), os.path.join(run_dir, "keyframes.json"),
                      node_ids, pipeline.memory, pipeline.graph, final_poses, T_grav_cam0,
                      result.loop_events, result.proximity_events, result.status_log)

    # -- report --
    report = build_trajectory_report(node_ids, final_poses, T_grav_cam0, align.aligned, align.reason,
                                      result.loop_events, result.proximity_events,
                                      result.cross_session_merges, result.status_log,
                                      n_frames_skipped_in_reconstruction=n_skipped)
    with open(os.path.join(run_dir, "trajectory_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)

    return report, T_grav_cam0
