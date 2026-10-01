"""
Section 11's acceptance table checks POSITION accuracy (tape-measured
floor markers). It has never checked whether the reported UNCERTAINTY
(cov_xy) is itself honest -- and an overconfident tracker can pass a
mean-error check while silently causing exactly the failure mode found
by testing this codebase against a physically-realistic D435i error
model: with the extrinsic-uncertainty gap this module's fixes address,
1 degree of calibration error produced ~14cm of real error against a
covariance that claimed ~1cm (NEES ~90, where a consistent filter
should read ~1.4 for a 2-DOF position update). That gap is invisible to
a plain accuracy check and would have been caught immediately by this
one.

Normalized Estimation Error Squared (NEES): for a track's reported
position error against ground truth, e = p_hat - p_true, and its
reported covariance P, NEES = e^T P^-1 e. For a consistent (honestly
uncertain) 2D estimator, NEES should average to 2 (the state
dimension) -- for a POSITION-only measurement update like the ones this
system reports (2-DOF position, not the full 4-DOF filter state), the
right reference is a chi-square(2) mean of 2, but position error after
a Kalman update is correlated with the prior in a way that makes the
textbook "average NEES == dof" only approximately right in practice;
1.0-3.0 is a reasonable non-degenerate band for a well-tuned system,
and importantly LOW cannot mean "better" -- it means the covariance is
padded so loose it no longer discriminates a good measurement from a
bad one, defeating the chi-square gate's whole purpose. The purpose of
this check is chiefly to catch the "median NEES >> 3" case (dangerous
overconfidence, e.g. a re-introduced omission of extrinsic uncertainty)
in CI, without needing hardware.

Run standalone: `python -m poi_localization.eval.consistency --deg-err 1.0`
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from poi_localization import geometry
from poi_localization.config import M4Config
from poi_localization.frames.types import FloorFrameTransform
from poi_localization.measurement.depth_measurement import compute_depth_measurement
from poi_localization.measurement.raycast_measurement import compute_raycast_measurement
from poi_localization.tracking.track_manager import DetectionMeasurements, TrackManager
from poi_perception.contracts import Intrinsics

PathFn = Callable[[float], Tuple[float, float]]  # t -> (x, y) ground-truth floor position


@dataclass
class ConsistencyResult:
    n_samples: int
    nees_median: float
    nees_p90: float
    position_error_median_m: float
    position_error_p90_m: float


def straight_line_path(x0: float = 1.0, y0: float = 0.6, speed_mps: float = 1.3) -> PathFn:
    return lambda t: (x0 + speed_mps * t, y0)


def _project(p_floor: np.ndarray, T: FloorFrameTransform, intr: Intrinsics) -> Optional[Tuple[float, float]]:
    p_cam = T.R.T @ (p_floor - T.t)
    if p_cam[2] <= 0.05:
        return None
    return (intr.fx * p_cam[0] / p_cam[2] + intr.cx, intr.fy * p_cam[1] / p_cam[2] + intr.cy)


def run_consistency_check(
    cfg: Optional[M4Config] = None,
    intr: Optional[Intrinsics] = None,
    camera_height_m: float = 2.3,
    camera_pitch_deg: float = 25.0,
    extrinsic_pitch_err_deg: float = 0.0,
    extrinsic_height_err_m: float = 0.0,
    sigma_rot_deg: float = 0.0,
    sigma_trans_m: float = 0.0,
    path_fn: Optional[PathFn] = None,
    n_frames: int = 300,
    dt: float = 1.0 / 30.0,
    seed: int = 7,
    gait: bool = True,
) -> ConsistencyResult:
    """Drives the REAL measurement + tracking code (not a re-implementation
    of it) against a synthetic walker with a physically-motivated sensor
    error model, and reports how honest the resulting cov_xy is via
    NEES. `sigma_rot_deg`/`sigma_trans_m` are what the system is TOLD to
    believe about its own calibration (FloorFrameTransform.sigma_*);
    `extrinsic_*_err_*` is the calibration error that ACTUALLY exists.
    Setting them equal simulates "the system's stated uncertainty
    matches reality"; leaving sigma_* at 0 with a nonzero err_*
    reproduces the pre-fix bug (extrinsic error present, but unmodeled).
    """
    cfg = cfg or M4Config.default()
    intr = intr or Intrinsics(fx=606.75, fy=606.57, cx=320.19, cy=237.06, width=640, height=480, depth_scale=0.001, baseline=0.05)
    path_fn = path_fn or straight_line_path()
    rng = np.random.default_rng(seed)

    T_true = FloorFrameTransform.from_height_and_yaw(camera_height_m, camera_pitch_deg, 0.0)
    T_est = FloorFrameTransform.from_height_and_yaw(
        camera_height_m + extrinsic_height_err_m,
        camera_pitch_deg + extrinsic_pitch_err_deg,
        0.0,
        sigma_rot_rad=np.radians(sigma_rot_deg),
        sigma_trans_m=sigma_trans_m,
    )
    tm = TrackManager(cfg)

    errors: List[float] = []
    nees_values: List[float] = []

    for k in range(n_frames):
        t = k * dt
        x, y = path_fn(t)
        gp = np.array([x, y, 0.0])
        torso = np.array([x, y, 1.25])

        px_torso = _project(torso, T_true, intr)
        if px_torso is None:
            continue
        p_cam_true = T_true.R.T @ (torso - T_true.t)
        z_true = p_cam_true[2]

        phase = np.sin(2 * np.pi * 0.9 * t)
        both_ankles = rng.random() > 0.25
        ankle_gt = gp + (np.array([0.35 * phase, 0.0, 0.0]) if (gait and not both_ankles) else 0.0)

        # depth measurement: front-surface + realistic D435i noise
        z_meas = (z_true - 0.12) + rng.normal(0, 0.015 * z_true + 0.005) + 0.008 * z_true
        u_n, v_n = px_torso[0] + rng.normal(0, 1.5), px_torso[1] + rng.normal(0, 1.5)
        depth_raw = _synthetic_depth_patch(u_n, v_n, z_meas, intr)
        poly = _patch_polygon(u_n, v_n)
        depth_m = compute_depth_measurement(depth_raw, poly, "full", intr, T_est, cfg.depth_measurement)

        px_ankle = _project(ankle_gt, T_true, intr)
        raycast_m = None
        if px_ankle is not None:
            noisy = (px_ankle[0] + rng.normal(0, 2.0), px_ankle[1] + rng.normal(0, 2.0))
            raycast_m = compute_raycast_measurement(
                noisy, "ankles" if both_ankles else "ankle_single", intr, T_est, cfg.raycast_measurement
            )

        if depth_m is None and raycast_m is None:
            continue

        det = DetectionMeasurements(1, 0.9, (0, 0, 10, 10), depth_m, raycast_m, 1.7)
        outputs, _ = tm.update(t, dt, [det])
        for o in outputs:
            if o.state != "confirmed":
                continue
            e = np.array(o.position_xy) - gp[:2]
            errors.append(float(np.linalg.norm(e)))
            nees_values.append(geometry.mahalanobis_sq(e, geometry.cov_upper_to_matrix(o.cov_xy)))

    errors_arr = np.array(errors) if errors else np.array([np.nan])
    nees_arr = np.array(nees_values) if nees_values else np.array([np.nan])
    return ConsistencyResult(
        n_samples=len(errors),
        nees_median=float(np.median(nees_arr)),
        nees_p90=float(np.percentile(nees_arr, 90)),
        position_error_median_m=float(np.median(errors_arr)),
        position_error_p90_m=float(np.percentile(errors_arr, 90)),
    )


def _synthetic_depth_patch(u: float, v: float, depth_m: float, intr: Intrinsics, half_w: int = 12, half_h: int = 28) -> np.ndarray:
    depth_raw = np.zeros((intr.height, intr.width), dtype=np.uint16)
    x0, x1 = max(0, int(u) - half_w), min(intr.width, int(u) + half_w)
    y0, y1 = max(0, int(v) - half_h), min(intr.height, int(v) + half_h)
    depth_raw[y0:y1, x0:x1] = int(round(depth_m / intr.depth_scale))
    return depth_raw


def _patch_polygon(u: float, v: float, half_w: int = 12, half_h: int = 28) -> List[Tuple[float, float]]:
    return [(u - half_w, v - half_h), (u + half_w, v - half_h), (u + half_w, v + half_h), (u - half_w, v + half_h)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deg-err", type=float, default=0.0, help="extrinsic pitch error to inject, degrees")
    ap.add_argument("--sigma-rot-deg", type=float, default=0.0, help="sigma_rot the system is TOLD to believe")
    ap.add_argument("--sigma-trans-m", type=float, default=0.0)
    ap.add_argument("--nees-max", type=float, default=3.0, help="fail if median NEES exceeds this")
    args = ap.parse_args(argv)

    result = run_consistency_check(
        extrinsic_pitch_err_deg=args.deg_err,
        sigma_rot_deg=args.sigma_rot_deg,
        sigma_trans_m=args.sigma_trans_m,
    )
    print(
        f"n={result.n_samples}  nees_median={result.nees_median:.2f}  nees_p90={result.nees_p90:.2f}  "
        f"pos_err_median_cm={result.position_error_median_m*100:.1f}  pos_err_p90_cm={result.position_error_p90_m*100:.1f}"
    )
    if result.nees_median > args.nees_max:
        print(f"FAIL: median NEES {result.nees_median:.2f} exceeds --nees-max {args.nees_max} -- covariance is overconfident.")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
