#!/usr/bin/env python3
"""
Run M3+M4 live.

    python3 scripts/run_m4.py --m3-config configs/m3.example.json \\
        --m4-config configs/m4.phaseA.example.json --source realsense

    # Phase A, no live calibration (uses fixed_transform from config):
    python3 scripts/run_m4.py --m3-config configs/m3.example.json \\
        --m4-config configs/m4.phaseA.fixed.example.json --source realsense

    # Phase B, once Module 2's calibration exists -- same command, just
    # point --m4-config at configs/m4.phaseB.example.json. No code changes.
    python3 scripts/run_m4.py --m3-config configs/m3.example.json \\
        --m4-config configs/m4.phaseB.example.json --source realsense
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poi_localization.runtime.m4_daemon import main

if __name__ == "__main__":
    main()
