"""
WP-T4: bird's-eye (top-down X-Y) and height (Z vs distance/time) plots
of a run's trajectory, in the gravity-aligned W_grav frame (see
pyslam.core.gravity_frame). Deliberately plain matplotlib, no
interactive/3D dependency -- these are meant to be opened as static
PNGs from a run directory, same spirit as map.ply.

Both functions degrade gracefully (skip, log a reason, don't raise) if
there's nothing to plot (e.g. static_60s's single keyframe) -- a
plotting failure must never take down the rest of run_slam.py/run_bag.py's
finally-block bookkeeping (see the Phase 1 plan's finding F7 on why that
block is careful about this already).
"""
from __future__ import annotations
import numpy as np

from pyslam.core.log import get_logger

log = get_logger("tools.plots")


def plot_bird_eye(path_png: str, odom_pts_grav: np.ndarray, final_pts_grav: np.ndarray,
                   loop_pairs_xy: list, proximity_pairs_xy: list, title: str = "Bird's-eye view (X-Y, gravity-aligned)") -> bool:
    """odom_pts_grav / final_pts_grav: (N,3) keyframe positions in
    W_grav, already time-ordered. loop_pairs_xy / proximity_pairs_xy:
    list of ((x1,y1),(x2,y2)) segments to draw as loop/proximity links.
    Returns False (and logs why) instead of raising if there's nothing
    plottable."""
    if final_pts_grav.shape[0] < 2:
        log.warning(f"plot_bird_eye: fewer than 2 keyframes, skipping ({path_png})")
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    if odom_pts_grav.shape[0] >= 2:
        ax.plot(odom_pts_grav[:, 0], odom_pts_grav[:, 1], "--", color="tab:gray",
                linewidth=1.0, label="odometry (uncorrected)", alpha=0.7)
    ax.plot(final_pts_grav[:, 0], final_pts_grav[:, 1], "-", color="tab:blue",
            linewidth=1.5, label="final (optimised)")
    ax.scatter(final_pts_grav[0, 0], final_pts_grav[0, 1], marker="o", color="green",
               s=80, zorder=5, label="start")
    ax.scatter(final_pts_grav[-1, 0], final_pts_grav[-1, 1], marker="s", color="red",
               s=80, zorder=5, label="end")
    for (x1, y1), (x2, y2) in proximity_pairs_xy:
        ax.plot([x1, x2], [y1, y2], "-", color="tab:orange", linewidth=0.8, alpha=0.6)
    for (x1, y1), (x2, y2) in loop_pairs_xy:
        ax.plot([x1, x2], [y1, y2], "-", color="tab:purple", linewidth=1.2, alpha=0.8)
    if loop_pairs_xy:
        ax.plot([], [], "-", color="tab:purple", linewidth=1.2, label="loop closure")
    if proximity_pairs_xy:
        ax.plot([], [], "-", color="tab:orange", linewidth=0.8, label="proximity link")

    ax.set_xlabel("x_grav (m)")
    ax.set_ylabel("y_grav (m)")
    ax.set_title(title)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path_png, dpi=150)
    plt.close(fig)
    return True


def plot_height(path_png: str, cumulative_dist_m: np.ndarray, height_m: np.ndarray,
                 t_s: np.ndarray, title: str = "Height (gravity-aligned Z)") -> bool:
    """Two panels: height vs cumulative path distance, height vs time --
    both are useful (distance shows spatial height structure e.g. going
    up a ramp; time shows temporal behaviour e.g. drift during a static
    hold)."""
    if height_m.shape[0] < 2:
        log.warning(f"plot_height: fewer than 2 keyframes, skipping ({path_png})")
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(cumulative_dist_m, height_m, "-o", color="tab:blue", markersize=2)
    ax1.set_xlabel("cumulative path distance (m)")
    ax1.set_ylabel("z_grav / height (m)")
    ax1.set_title("Height vs distance travelled")
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_s, height_m, "-o", color="tab:blue", markersize=2)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("z_grav / height (m)")
    ax2.set_title("Height vs time")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path_png, dpi=150)
    plt.close(fig)
    return True


def export_plots(run_dir: str, pipeline, result, T_grav_cam0: np.ndarray) -> None:
    """Called by run_slam.py/run_bag.py/run_synth.py right after
    trajectory_export.export_all(). Builds the (odom, final) keyframe
    point arrays in W_grav and the loop/proximity segment lists from the
    same data trajectory_export already assembled, so this stays a thin
    wrapper rather than re-deriving anything."""
    import os
    from pyslam.core import lie
    from pyslam.core.gravity_frame import pose_to_grav

    final_poses = result.final_poses or {}
    node_ids = sorted(nid for nid in pipeline.memory.all_node_ids() if nid in final_poses)
    if len(node_ids) < 2:
        log.warning("export_plots: fewer than 2 keyframes in final_poses, skipping all plots")
        return

    odom_pts, final_pts, t_list = [], [], []
    for nid in node_ids:
        node = pipeline.memory.get(nid)
        odom_pts.append(pose_to_grav(node.pose_odom, T_grav_cam0)[:3, 3])
        final_pts.append(pose_to_grav(final_poses[nid], T_grav_cam0)[:3, 3])
        t_list.append(node.sig.t)
    odom_pts = np.array(odom_pts)
    final_pts = np.array(final_pts)
    t_arr = np.array(t_list)

    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    def _segments(events):
        segs = []
        for a, b, link in events:
            if a in id_to_idx and b in id_to_idx:
                pa = final_pts[id_to_idx[a]][:2]
                pb = final_pts[id_to_idx[b]][:2]
                segs.append((tuple(pa), tuple(pb)))
        return segs

    loop_segs = _segments(result.loop_events)
    prox_segs = _segments(result.proximity_events)

    plot_bird_eye(os.path.join(run_dir, "trajectory_bird_eye.png"),
                  odom_pts, final_pts, loop_segs, prox_segs)

    cumulative = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(final_pts, axis=0), axis=1))])
    # WP-K1: an identity T_grav_cam0 means gravity alignment failed (see
    # gravity_frame.GravityAlignmentResult.aligned); the "height" axis is
    # then cam0's optical Z, not vertical -- say so ON the plot, since
    # the PNG is often looked at without the JSON report beside it.
    gravity_ok = not np.allclose(T_grav_cam0, np.eye(4))
    plot_height(os.path.join(run_dir, "trajectory_height.png"),
                cumulative, final_pts[:, 2], t_arr - t_arr[0],
                title=("Height (gravity-aligned Z)" if gravity_ok else
                       "NOT HEIGHT -- gravity alignment failed; Z = first camera's optical axis"))
