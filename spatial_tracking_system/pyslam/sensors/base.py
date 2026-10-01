"""SensorSource is defined in pyslam.core.types (the frozen contract).
This module just re-exports it so `from pyslam.sensors.base import SensorSource`
reads naturally alongside realsense.py / synthetic.py / bagfile.py."""
from pyslam.core.types import SensorSource, Frame, Intrinsics  # noqa: F401
