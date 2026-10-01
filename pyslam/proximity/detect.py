"""
WP-P4 proximity detection (architecture doc section 5, P4): "local-space
and local-time proximity detection -- links nodes that are
geometrically close even without appearance match... this is what fixes
corridors and reverse traversals," where the SAME physical location is
viewed from a very different angle (a corridor walked outbound then
back), which can defeat appearance-based retrieval (ORB matching needs
enough viewpoint overlap) even though the poses are, by the graph's own
current estimate, right next to each other.

Deliberately geometry-only for candidate GENERATION (no appearance
signal at all -- that's the point, it's the complement to retrieval,
not a variant of it), but every candidate still goes through the SAME
`GeometricVerifier.verify()` used for appearance-triggered loop closure
before a link is accepted. Proximity detection widens WHICH pairs get
proposed for geometric verification; it does not relax what counts as
a verified link.
"""
from __future__ import annotations
import numpy as np

from pyslam.memory.memory import Memory


def find_proximity_candidates(memory: Memory, existing_pairs: set[tuple[int, int]],
                               radius_m: float, min_index_gap: int) -> list[tuple[int, int]]:
    """Pairs of WM-resident nodes whose CURRENT pose_map positions are
    within `radius_m`, excluding pairs already linked (existing_pairs,
    normalised (min,max) tuples -- odom links, prior loop links, AND
    prior proximity links all belong in this set so the same pair isn't
    proposed twice) and pairs too close in creation-index to be a
    meaningful revisit (min_index_gap, same purpose as WP-P3's gate
    tests' own true-pair definition -- excludes trivially-adjacent
    keyframes, not a genuine spatial return).

    O(|WM|^2): fine at this project's tested WM scale (low hundreds to
    ~1000); would need a spatial index (k-d tree / grid) before this
    stops being fine, same caveat WP-P3's incremental vocabulary has
    about its own O(N*V) scaling -- not attempted here, out of P4's
    stated scope.
    """
    ids = sorted(memory.working_set())
    positions = {i: memory.get(i).pose_map[:3, 3] for i in ids}
    out = []
    for ia, a in enumerate(ids):
        for b in ids[ia + 1:]:
            if b - a <= min_index_gap:
                continue
            pair = (a, b)
            if pair in existing_pairs:
                continue
            if np.linalg.norm(positions[a] - positions[b]) <= radius_m:
                out.append(pair)
    return out
