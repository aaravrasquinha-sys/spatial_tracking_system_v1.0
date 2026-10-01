"""
WP-LIVE: the "site frame", replacing the original plan's single-room
"room frame" (z=0 floor, one rectangular interior). Approved direction
is "generalize on any indoor space" -- multiple levels, ramps, open-plan
areas, corridors.

Definition:
  - The REFERENCE support plane is the support-kind PlaneLandmark
    (pyslam/mapping/planes.py) nearest the first keyframe's camera
    viewpoint -- i.e. whatever the camera was standing/mounted above
    when mapping started. This is deliberately viewpoint-relative, not
    "the floor of the room": on a different level, the reference is
    that level's own local surface.
  - site.z = 0 on the reference plane, +z along its normal (oriented
    toward the camera side, i.e. "up" from the plane).
  - site.origin = the first keyframe's camera centre, projected onto
    the reference plane along its normal.
  - site.+x = the first keyframe's own forward direction, projected
    onto the reference plane (same convention W_grav already uses in
    pyslam/core/gravity_frame.py, so plots/exports look the same shape
    of thing whether or not this module's richer frame is available).

Other support planes (a landing on a different level, a ramp, a
tabletop) are NOT forced into site.z=0 -- they keep whatever height
they actually have in the site frame. pyslam/mapping/layers.py's height
field is what carries "the floor here is at 0.4m" information forward
to M4, not this module pretending there's only one floor.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

from pyslam.core import lie
from pyslam.mapping.planes import PlaneLandmark, PLANE_KIND_SUPPORT


class SiteFrame:
    def __init__(self, T_site_cam0: np.ndarray, reference_plane_id: Optional[int]):
        self.T_site_cam0 = T_site_cam0          # cam0 (graph's native world) -> site
        self.reference_plane_id = reference_plane_id

    def pose_to_site(self, T_cam0_x: np.ndarray) -> np.ndarray:
        return self.T_site_cam0 @ T_cam0_x

    def point_to_site(self, p_cam0: np.ndarray) -> np.ndarray:
        return lie.transform_points(self.T_site_cam0, np.atleast_2d(p_cam0))


def build_site_frame(first_keyframe_pose_cam0: np.ndarray,
                      reference_plane: Optional[PlaneLandmark],
                      fallback_up_cam0: Optional[np.ndarray] = None,
                      cam0_forward_cam: np.ndarray = np.array([0.0, 0.0, 1.0])) -> SiteFrame:
    """first_keyframe_pose_cam0: T_cam0_kf0 (should be identity by
    construction -- kf0 IS cam0's origin -- but passed explicitly so
    this function has no hidden global-state dependency).

    reference_plane: a PlaneLandmark already expressed in the CAM0
    frame (kind==support), or None if no support plane has been found
    yet (e.g. the first few keyframes of a run before enough floor is
    visible) -- in that case fallback_up_cam0 (e.g. gravity from a
    quasi-static IMU window, same source pyslam/core/gravity_frame.py
    already uses) provides orientation only, with z=0 pinned at the
    camera's own height rather than a fitted plane; SiteFrame callers
    MUST check reference_plane_id is not None before trusting the
    site's z=0 as a real floor height (mirrors the honesty discipline
    gravity_frame.py's `aligned` flag already establishes for W_grav).
    """
    R_cam0_kf0 = first_keyframe_pose_cam0[:3, :3]
    t_cam0_kf0 = first_keyframe_pose_cam0[:3, 3]

    if reference_plane is not None:
        normal = reference_plane.normal_site / max(np.linalg.norm(reference_plane.normal_site), 1e-12)
        # ensure the normal points toward the camera side (offset convention:
        # normal . x = offset; camera above the plane means normal . t_cam0_kf0 > offset)
        if normal @ t_cam0_kf0 - reference_plane.offset_site < 0:
            normal = -normal
        up = normal
        plane_id = reference_plane.id
        # origin: project the camera centre onto the plane along its normal
        d = float(normal @ t_cam0_kf0 - reference_plane.offset_site)
        origin = t_cam0_kf0 - d * normal
    else:
        up = (fallback_up_cam0 / max(np.linalg.norm(fallback_up_cam0), 1e-12)
              if fallback_up_cam0 is not None else np.array([0.0, 1.0, 0.0]))
        plane_id = None
        origin = t_cam0_kf0.copy()

    fwd_world = R_cam0_kf0 @ cam0_forward_cam
    fwd = fwd_world - (fwd_world @ up) * up
    fwd_norm = np.linalg.norm(fwd)
    if fwd_norm < 1e-3:
        for fallback in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])):
            fwd = fallback - (fallback @ up) * up
            fwd_norm = np.linalg.norm(fwd)
            if fwd_norm >= 1e-3:
                break
    x_site = fwd / fwd_norm
    z_site = up
    y_site = np.cross(z_site, x_site)
    y_site = y_site / max(np.linalg.norm(y_site), 1e-12)
    x_site = np.cross(y_site, z_site)

    R_cam0_site = np.stack([x_site, y_site, z_site], axis=1)
    R_site_cam0 = R_cam0_site.T
    t_site_cam0 = -R_site_cam0 @ origin
    T_site_cam0 = lie.make_T(R_site_cam0, t_site_cam0)
    return SiteFrame(T_site_cam0, plane_id)
