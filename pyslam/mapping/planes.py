"""
WP-LIVE N4/site-frame: plane landmarks, generalised.

The original room-scale plan had ONE floor plane at z=0 and wall-
squareness as an acceptance check. Approved direction is "generalize
on any indoor space" -- this module extracts and tracks an arbitrary
number of planar surfaces per keyframe (floors/landings/ramps at any
height, walls at any angle, overhead surfaces), classifies each by its
relationship to gravity rather than to a fixed global frame, and
associates repeat observations of the "same" physical plane across
keyframes into a landmark. It deliberately never assumes orthogonality
between planes -- see planning notes: "Never force walls to be
orthogonal."

Two independent pieces, kept separate on purpose (single-responsibility,
and each is independently gate-testable):
  - fit_planes(): pure geometry, greedy multi-plane RANSAC over a point
    cloud with normals. No notion of "floor" or "wall" here.
  - classify_plane() / associate_plane(): interpretation, using a given
    up-direction (gravity, in camera frame) purely as a LABEL, never as
    a constraint on the fit itself.

Consumed by:
  - pyslam/mapping/site_frame.py (the support plane nearest the
    reference viewpoint defines the site frame's z=0 and orientation)
  - the backend process's OrientedPlane3 factors (WP-LIVE graph/
    backend_isam2.py), when GTSAM is available
  - pyslam/mapping/layers.py's per-plane height field
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class PlaneCandidate:
    normal: np.ndarray      # unit 3-vector, camera frame, sign TOWARD the camera-side half-space
    offset: float           # plane: normal . x = offset  (offset = normal . point_on_plane)
    inlier_idx: np.ndarray  # indices into the input point array
    rms_m: float


def _fit_plane_svd(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares plane through pts (N,3): normal via SVD of the
    centred points (smallest singular vector), offset = normal.centroid.
    N must be >= 3."""
    centroid = pts.mean(axis=0)
    centred = pts - centroid
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    normal = vt[-1]
    normal = normal / max(np.linalg.norm(normal), 1e-12)
    offset = float(normal @ centroid)
    return normal, offset


def fit_planes(pts: np.ndarray, dist_thresh_m: float = 0.02, min_inliers: int = 300,
               max_planes: int = 6, ransac_iters: int = 200,
               rng: Optional[np.random.Generator] = None) -> list[PlaneCandidate]:
    """Greedy multi-plane RANSAC: fit the best plane, remove its
    inliers, repeat. Stops at max_planes or when no candidate clears
    min_inliers. Pure geometry -- no up-vector, no classification, so
    this works identically on a floor, a wall, a ramp, or a tabletop.
    O(max_planes * ransac_iters * N) -- keep pts down-sampled by the
    caller (a keyframe's fused-depth points at a coarse stride) to stay
    cheap enough for the backend's per-keyframe budget.
    """
    rng = rng or np.random.default_rng(0)
    remaining_idx = np.arange(pts.shape[0])
    out: list[PlaneCandidate] = []
    work = pts.copy()

    while len(out) < max_planes and work.shape[0] >= max(min_inliers, 3):
        best_inliers = None
        best_normal = best_offset = None
        n = work.shape[0]
        for _ in range(ransac_iters):
            idx3 = rng.choice(n, size=3, replace=False)
            p0, p1, p2 = work[idx3]
            v1, v2 = p1 - p0, p2 - p0
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-9:
                continue
            normal = normal / norm
            offset = float(normal @ p0)
            dist = np.abs(work @ normal - offset)
            inliers = np.where(dist < dist_thresh_m)[0]
            if best_inliers is None or inliers.size > best_inliers.size:
                best_inliers, best_normal, best_offset = inliers, normal, offset

        if best_inliers is None or best_inliers.size < min_inliers:
            break

        # refine with a real least-squares fit on the inlier set
        normal, offset = _fit_plane_svd(work[best_inliers])
        dist = np.abs(work @ normal - offset)
        refined_inliers = np.where(dist < dist_thresh_m)[0]
        if refined_inliers.size < min_inliers:
            break
        normal, offset = _fit_plane_svd(work[refined_inliers])
        rms = float(np.sqrt(np.mean((work[refined_inliers] @ normal - offset) ** 2)))

        out.append(PlaneCandidate(
            normal=normal, offset=offset,
            inlier_idx=remaining_idx[refined_inliers], rms_m=rms,
        ))

        mask = np.ones(work.shape[0], dtype=bool)
        mask[refined_inliers] = False
        work = work[mask]
        remaining_idx = remaining_idx[mask]

    return out


PLANE_KIND_SUPPORT = "support"
PLANE_KIND_VERTICAL = "vertical"
PLANE_KIND_OVERHEAD = "overhead"
PLANE_KIND_OTHER = "other"


def classify_plane(normal_cam: np.ndarray, centroid_cam: np.ndarray, up_cam: np.ndarray,
                    camera_height_hint_m: Optional[float] = None,
                    angle_tol_deg: float = 20.0) -> str:
    """Labels a plane using ONLY its own geometry + the given up
    direction (camera-frame gravity, e.g. from a quasi-static IMU
    window) -- never a global room assumption. A plane whose normal is
    within angle_tol_deg of vertical (up or down) and lies BELOW the
    camera (in the up direction) is "support" -- a floor, a landing, a
    stair tread, a tabletop, all treated the same way, at whatever
    height they're actually at. One near-vertical ABOVE the camera is
    "overhead" (a ceiling, a low soffit). Anything with normal roughly
    perpendicular to up is "vertical" (a wall, at any yaw -- no
    orthogonality assumed between two vertical planes). Everything else
    (sloped, e.g. a ramp near the tolerance boundary) is "other" and
    still gets tracked as a landmark, just not treated as a walking
    surface by site_frame.py.
    """
    normal = normal_cam / max(np.linalg.norm(normal_cam), 1e-12)
    up = up_cam / max(np.linalg.norm(up_cam), 1e-12)
    cos_up = float(np.clip(normal @ up, -1.0, 1.0))
    angle_from_up = np.degrees(np.arccos(abs(cos_up)))

    if angle_from_up > (90.0 - angle_tol_deg):
        return PLANE_KIND_VERTICAL
    # near-horizontal: is it below or above the camera? "below" means
    # walking toward it along -up decreases height, i.e. the plane sits
    # on the far side of the camera from "up".
    rel = float((centroid_cam - np.zeros(3)) @ up)  # camera is origin in its own frame
    if rel < 0:
        return PLANE_KIND_SUPPORT
    return PLANE_KIND_OVERHEAD


@dataclass
class PlaneLandmark:
    id: int
    normal_site: np.ndarray   # unit 3-vector, in the SITE frame (stable across keyframes)
    offset_site: float        # site-frame plane equation: normal . x = offset
    kind: str
    n_observations: int = 0
    last_seen_kf: int = -1
    extent_site: Optional[np.ndarray] = None  # (min_xy, max_xy) axis-aligned bound in the
        # plane's own local 2D coords, for cheap overlap tests -- optional, filled by caller


def associate_plane(existing: list[PlaneLandmark], normal_site: np.ndarray, offset_site: float,
                     kind: str, angle_thresh_deg: float = 8.0, offset_thresh_m: float = 0.08
                     ) -> Optional[int]:
    """Returns the id of the best-matching existing landmark this
    observation should be folded into, or None if it looks new.
    Matching is by (normal angle, offset) only -- no spatial-overlap
    check here (callers with real 2D extent, e.g. the backend process,
    should additionally require bounding-region overlap before
    accepting a match, to avoid merging two coplanar-but-disjoint walls
    in different rooms; kept out of this pure function so it stays
    unit-testable without extent data)."""
    normal_site = normal_site / max(np.linalg.norm(normal_site), 1e-12)
    best_id, best_score = None, None
    for lm in existing:
        if lm.kind != kind:
            continue
        cos = float(np.clip(abs(normal_site @ lm.normal_site), -1.0, 1.0))
        angle = np.degrees(np.arccos(cos))
        if angle > angle_thresh_deg:
            continue
        # sign-agnostic offset comparison (normal sign can flip between
        # observations of the same physical surface)
        d = min(abs(offset_site - lm.offset_site), abs(offset_site + lm.offset_site))
        if d > offset_thresh_m:
            continue
        score = angle / max(angle_thresh_deg, 1e-6) + d / max(offset_thresh_m, 1e-6)
        if best_score is None or score < best_score:
            best_score, best_id = score, lm.id
    return best_id
