"""R1 -- map-anchored identity recovery ("AnchorID").

Runtime identity code lives here, inside M4. Import boundaries (enforced by
tests/integration/test_boundaries.py): this package may import poi_perception contracts only --
never pyslam, poi_present or sts. Identity is OFF by default (`M4Config.identity.enabled=False`)
and, when off, M4 behaves bit-for-bit as before.

Keep this file import-free: poi_localization.config imports identity.config, so importing the
engine here would create a cycle.
"""
