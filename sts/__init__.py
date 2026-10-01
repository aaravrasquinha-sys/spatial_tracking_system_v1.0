"""sts -- the orchestration layer of the Spatial Tracking System.

This package is the ONLY place that knows about every module at once.
It contains no perception, SLAM, localization or presentation maths: it
wires the existing modules together through their published contracts,
validates that the wiring is consistent, and supervises the live run.

    pyslam            M1 (map) + M2 (anchor)       -- never imports sts
    poi_perception    M3                           -- never imports sts
    poi_localization  M4                           -- never imports sts
    poi_present       M5                           -- never imports sts
    sts               M6 (runtime, health, replay) -- imports all of them

See docs/ARCHITECTURE.md and contracts/ for the rules that keep it so.
"""
__version__ = "1.0.0"
