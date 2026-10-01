#!/usr/bin/env python3
"""
Export a .pt pose checkpoint to a TensorRT engine, on-device (Section 3).

    python3 scripts/export_model.py --weights yolo11n-pose.pt \\
        --out models/engines/yolo11n_pose_fp16.engine --width 640 --height 480
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poi_perception.models.export_engine import main

if __name__ == "__main__":
    main()
