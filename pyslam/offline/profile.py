"""
The settings the offline mapper is built around, in one place.

They are the settings of the run that scored best on room2 (R4: RMSE 0.50 m, start-to-end gap
1.24 m, against 1.28 m / 3.46 m for the original defaults) and were then used by the offline
mapper (run_offline.py). They are NOT the live defaults: Config() keeps its old values so the live
Jetson chain and the existing test gates behave exactly as before. run_map.py applies this profile
automatically; to see or change a single value use --config-override KEY=VALUE as usual.

  bridge_mode=gyro                 across a lost-tracking gap, take the rotation from the gyro (not zero)
  gyro_aid_enabled=true            use the gyro to catch and replace wrong camera rotations
  gyro_aid_verify_samples=100      the gyro-vs-camera self-check looks at 100 well-tracked frames ...
  gyro_aid_verify_max_median_deg=3.0   ... and accepts up to 3 degrees median difference (gyro drift over
                                   a keyframe window is ~2 degrees; a wrong mounting angle is tens)
  gyro_aid_coast=false             on lost frames do NOT keep moving at the last speed (it ran away:
                                   +1 m error at the bare wall in run R3)
  proximity_enabled=false          the live proximity search is slow offline and run_offline.py does a
                                   better one over all keyframes
"""
from __future__ import annotations
from typing import Iterable, Optional

from pyslam.core.config import Config, apply_overrides

OFFLINE_OVERRIDES = (
    "bridge_mode=gyro",
    "gyro_aid_enabled=true",
    "gyro_aid_verify_samples=100",
    "gyro_aid_verify_max_median_deg=3.0",
    "gyro_aid_coast=false",
    "proximity_enabled=false",
)


def offline_config(extra_overrides: Optional[Iterable[str]] = None, profile: str = "offline") -> Config:
    """Config for an offline run: the profile first, then the caller's own overrides (which win)."""
    if profile not in ("offline", "default"):
        raise ValueError(f"unknown profile {profile!r} (choose 'offline' or 'default')")
    items = list(OFFLINE_OVERRIDES) if profile == "offline" else []
    items += list(extra_overrides or [])
    return apply_overrides(Config(), items)
