#!/usr/bin/env python3
"""
Run M3 live.

    python3 scripts/run_m3.py --config configs/m3.example.json --source realsense
    python3 scripts/run_m3.py --config configs/m3.example.json --source synthetic --max-frames 300
    python3 scripts/run_m3.py --config configs/m3.example.json --source realsense --playback data/room_loop.bag
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poi_perception.runtime.m3_daemon import main

if __name__ == "__main__":
    main()
