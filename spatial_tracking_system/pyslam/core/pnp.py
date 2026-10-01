"""
Shared robust-PnP flag selection.

Phase 0's odometry.py computed this exact try/except but then ignored the
result, always calling solvePnPRansac with flags=cv2.SOLVEPNP_ITERATIVE.
verify.py didn't even attempt USAC. Centralised here so both call sites
actually use whatever the running OpenCV build supports, and so
env_probe.py's "USAC available" report matches what the pipeline does.

USAC_MAGSAC is preferred when available: RANSAC with LO refinement, no
manually-tuned threshold escalation, and (per the OpenCV 5.0 architecture
notes) the intended default for robust estimation going forward.
"""
from __future__ import annotations
import cv2

_flag = None


def robust_pnp_flag() -> int:
    global _flag
    if _flag is None:
        _flag = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.SOLVEPNP_ITERATIVE
    return _flag


def robust_pnp_flag_name() -> str:
    return "USAC_MAGSAC" if hasattr(cv2, "USAC_MAGSAC") and robust_pnp_flag() == cv2.USAC_MAGSAC else "SOLVEPNP_ITERATIVE"
