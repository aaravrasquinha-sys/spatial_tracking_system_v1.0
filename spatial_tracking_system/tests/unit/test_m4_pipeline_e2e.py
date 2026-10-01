import json

import numpy as np
import pytest

from poi_localization.config import M4Config
from poi_localization.frames.frame_provider import FixedFrameProvider
from poi_localization.frames.types import FloorFrameTransform
from poi_localization.io.world_track import WorldTrackPhaseA, WorldTrackPhaseB
from poi_localization.runtime.m4_pipeline import M4Pipeline
from poi_perception.contracts import Frame, Intrinsics
from poi_perception.io.detection2d import Detection2D

INTR = Intrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480, depth_scale=0.001, baseline=0.05)


def _make_person_detection(floor_transform, x, y, frame_id, track_id_2d=1):
    """Build a Detection2D + matching depth image for a standing person
    at known floor position (x, y), using the same forward-projection
    approach as the measurement-layer tests."""
    R_inv = floor_transform.R.T
    t_inv = -R_inv @ floor_transform.t

    def project(px, py, pz):
        p_cam = R_inv @ np.array([px, py, pz]) + t_inv
        u = INTR.fx * p_cam[0] / p_cam[2] + INTR.cx
        v = INTR.fy * p_cam[1] / p_cam[2] + INTR.cy
        return u, v, p_cam[2]

    # torso center ~1.0m up, ankles at floor level
    u_torso, v_torso, d_torso = project(x, y, 1.0)
    u_ankle_l, v_ankle_l, _ = project(x - 0.1, y, 0.02)
    u_ankle_r, v_ankle_r, _ = project(x + 0.1, y, 0.02)
    u_nose, v_nose, d_nose = project(x, y, 1.7)

    depth_raw = np.full((480, 640), 2000, dtype=np.uint16)  # background floor-ish depth
    half_w, half_h = 15, 30
    x0, x1 = int(u_torso - half_w), int(u_torso + half_w)
    y0, y1 = int(v_torso - half_h), int(v_torso + half_h)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(640, x1), min(480, y1)
    depth_raw[y0:y1, x0:x1] = int(round(d_torso / INTR.depth_scale))
    # small patch of valid depth at the head keypoint too (height estimator)
    hx0, hx1 = max(0, int(u_nose) - 3), min(640, int(u_nose) + 3)
    hy0, hy1 = max(0, int(v_nose) - 3), min(480, int(v_nose) + 3)
    depth_raw[hy0:hy1, hx0:hx1] = int(round(d_nose / INTR.depth_scale))

    torso_polygon = [(u_torso - half_w, v_torso - half_h), (u_torso + half_w, v_torso - half_h),
                      (u_torso + half_w, v_torso + half_h), (u_torso - half_w, v_torso + half_h)]
    footpoint = ((u_ankle_l + u_ankle_r) / 2, (v_ankle_l + v_ankle_r) / 2)

    keypoints = {
        "nose": (u_nose, v_nose, 0.9),
        "left_ankle": (u_ankle_l, v_ankle_l, 0.9),
        "right_ankle": (u_ankle_r, v_ankle_r, 0.9),
    }

    det = Detection2D(
        t_capture=0.0,
        frame_id=frame_id,
        cam_id="cam0",
        track_id_2d=track_id_2d,
        bbox=(u_torso - half_w, v_torso - half_h * 1.6, u_torso + half_w, v_torso + half_h * 1.6),
        det_conf=0.9,
        keypoints=keypoints,
        footpoint_px=footpoint,
        footpoint_source="ankles",
        torso_polygon_px=torso_polygon,
        torso_quality="full",
        low_confidence=False,
        track_state="tracked",
    )
    frame = Frame(t=frame_id * (1 / 30.0), rgb=np.zeros((480, 640, 3), dtype=np.uint8), depth=depth_raw, intr=INTR, frame_id=frame_id)
    return frame, det


def test_pipeline_phase_a_single_person_converges_near_true_position():
    cfg = M4Config.default()
    cfg.phase = "A"
    cfg.track.tentative_confirm_hits = 2
    ft = FloorFrameTransform.from_height_and_yaw(2.2, pitch_deg=20, yaw_deg=0)
    provider = FixedFrameProvider(ft)
    pipeline = M4Pipeline(cfg, provider)

    last_records = []
    for i in range(15):
        frame, det = _make_person_detection(ft, 2.0, 0.3, i)
        records = pipeline.process(frame, [det])
        if records:
            last_records = records

    assert len(last_records) == 1
    rec = last_records[0]
    assert isinstance(rec, WorldTrackPhaseA)
    assert rec.state == "confirmed"
    # true position (2.0, 0.3); body-thickness offset pushes it a little
    # further from the camera -- allow a few cm of slack for that plus
    # filter smoothing.
    assert abs(rec.p_local[0] - 2.0) < 0.15
    assert abs(rec.p_local[1] - 0.3) < 0.15
    assert rec.height_m is not None
    assert 1.4 < rec.height_m < 2.0


def test_pipeline_walking_track_keeps_stable_id_and_reasonable_velocity():
    cfg = M4Config.default()
    cfg.track.tentative_confirm_hits = 2
    ft = FloorFrameTransform.from_height_and_yaw(2.2, pitch_deg=20, yaw_deg=0)
    provider = FixedFrameProvider(ft)
    pipeline = M4Pipeline(cfg, provider)

    track_ids = set()
    speeds = []
    x = 1.0
    for i in range(40):
        x += 0.03  # ~0.9 m/s at 30fps
        frame, det = _make_person_detection(ft, x, 0.0, i)
        records = pipeline.process(frame, [det])
        for r in records:
            track_ids.add(r.world_track_id)
            speed = float(np.hypot(*r.v_local[:2]))
            speeds.append(speed)

    assert len(track_ids) == 1
    # after the filter has settled, speed should be in a sane walking range
    assert 0.3 < np.median(speeds[-10:]) < 1.5


def test_pipeline_phase_b_produces_poi_v1_shaped_records(tmp_path):
    # Build a physically sensible T_room_cam (camera at 2.2m, tilted down
    # 20deg) rather than hand-typing a rotation -- an arbitrary rotation
    # matrix (e.g. identity) doesn't represent a camera actually looking
    # out across the room, so points behind it would look "invalid" for
    # reasons that have nothing to do with the code under test.
    cam_pose = FloorFrameTransform.from_height_and_yaw(2.2, pitch_deg=20, yaw_deg=0, frame_name="room")
    T_room_cam = np.eye(4)
    T_room_cam[:3, :3] = cam_pose.R
    T_room_cam[:3, 3] = cam_pose.t
    calib = {
        "cam_id": "cam0",
        "map_id": "testmap123",
        "T_room_cam": T_room_cam.tolist(),
    }
    calib_path = tmp_path / "calib.json"
    calib_path.write_text(json.dumps(calib))

    cfg = M4Config.default()
    cfg.phase = "B"
    cfg.phase_b.calibration_path = str(calib_path)
    cfg.track.tentative_confirm_hits = 2

    from poi_localization.frames.frame_provider import build_frame_provider

    provider = build_frame_provider(cfg)
    pipeline = M4Pipeline(cfg, provider)

    ft = provider.get_transform()
    last_records = []
    for i in range(10):
        frame, det = _make_person_detection(ft, 1.5, 0.0, i)
        records = pipeline.process(frame, [det])
        if records:
            last_records = records

    assert len(last_records) == 1
    rec = last_records[0]
    assert isinstance(rec, WorldTrackPhaseB)
    assert rec.id == last_records[0].id
    assert len(rec.bbox) == 4
    assert rec.p[2] == 0.0
