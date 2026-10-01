"""
Section 2, Phase B: "Module 4's only change is: instead of expressing 3D
positions in the local floor frame, express them in the room frame...
T_room_cam replaces the Phase-A local-frame transform wholesale, since
both are 'some fixed floor-referenced frame relative to the camera.'"

Reads the calibration file shape documented in the full system plan's
Section 4.2 exactly:

    {
      "cam_id": "cam0",
      "map_id": "a3f9c2e1b7d4",
      "T_room_cam": [[...4x4...]],
      ...
    }

Module 4 doesn't validate the calibration itself (that's Module 2's
job, at build time) -- it only loads T_room_cam and map_id, and (per
Section 13) refuses to run if the map_id here disagrees with the one
the rest of the runtime expects, matching the full plan's M6 discipline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from poi_localization.frames.types import FloorFrameTransform
from poi_localization.log import get_logger

log = get_logger("frames.room_frame")


class CalibrationMapIdMismatch(RuntimeError):
    pass


def load_room_frame(
    calibration_path: str | Path,
    expected_map_id: Optional[str] = None,
    min_sigma_rot_deg: float = 0.05,
    min_sigma_trans_m: float = 0.003,
) -> FloorFrameTransform:
    raw = json.loads(Path(calibration_path).read_text())

    map_id = raw.get("map_id")
    if expected_map_id is not None and map_id is not None and map_id != expected_map_id:
        raise CalibrationMapIdMismatch(
            f"Calibration file's map_id ({map_id}) does not match the expected "
            f"map_id ({expected_map_id}) -- refusing to publish registered "
            f"positions against a mismatched map (full plan's M6 discipline). "
            f"Re-run Module 2's calibration against the currently loaded map, or "
            f"pass the correct expected_map_id."
        )

    T_room_cam = np.array(raw["T_room_cam"], dtype=np.float64)
    if T_room_cam.shape != (4, 4):
        raise ValueError(f"T_room_cam in {calibration_path} is not 4x4: shape {T_room_cam.shape}")

    R = T_room_cam[:3, :3]
    t = T_room_cam[:3, 3]

    # Sanity-check R is a proper rotation (orthonormal, det +1) -- catches
    # a hand-edited or malformed calibration file before it silently
    # corrupts every position downstream.
    should_be_identity = R @ R.T
    if not np.allclose(should_be_identity, np.eye(3), atol=1e-3):
        raise ValueError(
            f"T_room_cam's rotation part in {calibration_path} is not orthonormal "
            f"(R @ R.T != I) -- calibration file is malformed."
        )
    det = np.linalg.det(R)
    if det < 0:
        raise ValueError(
            f"T_room_cam's rotation part in {calibration_path} has determinant {det:.3f} "
            f"(expected +1) -- this is a reflection, not a rotation; calibration file is malformed."
        )

    # The calibration file's own "sigma" block (Section 4.2 of the full
    # plan: ICP RMSE, IMU/floor-fit cross-checks, remount repeatability
    # all feed this number at build time) is M2's honest estimate of how
    # well T_room_cam is actually known -- previously logged here and
    # then silently discarded, so every downstream position measurement
    # implicitly assumed a perfect calibration. Floors keep a missing or
    # implausibly-tiny sigma block from producing an overconfident
    # transform (Phase B should be BETTER than Phase A's fallback floor,
    # since it has an actual multi-source calibration behind it, but
    # never exactly zero).
    sigma_block = raw.get("sigma", {})
    sigma_rot_deg = max(min_sigma_rot_deg, float(sigma_block.get("rot_deg", min_sigma_rot_deg)))
    sigma_trans = max(min_sigma_trans_m, float(sigma_block.get("trans_m", min_sigma_trans_m)))
    if not sigma_block:
        log.warning(
            f"{calibration_path} has no 'sigma' block -- falling back to a "
            f"conservative floor (rot={min_sigma_rot_deg}deg, trans={min_sigma_trans_m}m). "
            f"M2 should always emit its own measured sigma; treat this calibration "
            f"file as suspect if this warning is unexpected."
        )

    transform = FloorFrameTransform(
        R=R, t=t, frame_name="room", map_id=map_id,
        sigma_rot_rad=float(np.radians(sigma_rot_deg)),
        sigma_trans_m=float(sigma_trans),
    )
    log.info(
        f"Loaded room frame from {calibration_path}: map_id={map_id}, "
        f"camera_height={transform.camera_height_m:.3f}m, "
        f"sigma_trans={transform.sigma_trans_m:.4f}m, "
        f"sigma_rot={np.degrees(transform.sigma_rot_rad):.3f}deg"
    )
    return transform
