"""3D joint lifting (poi_localization/measurement/joints.py) and its path to the wire."""
import json
from pathlib import Path

import jsonschema
import numpy as np

from poi_localization.config import JointsConfig, M4Config
from poi_localization.frames.frame_provider import FixedFrameProvider
from poi_localization.frames.types import FloorFrameTransform
from poi_localization.measurement.joints import lift_joints
from poi_localization.runtime.m4_pipeline import M4Pipeline
from poi_perception.contracts import COCO_KEYPOINT_NAMES, Frame
from poi_present.adapter import record_to_wire
from poi_present.schema import tracks_message
from tests.unit.test_m4_pipeline_e2e import INTR, _make_person_detection

FT = FloorFrameTransform.from_height_and_yaw(2.2, pitch_deg=20, yaw_deg=0)
R_INV = FT.R.T
T_INV = -R_INV @ FT.t


def _body(x, y, wave=0.0):
    """17 true joint positions (room frame, metres) for a person standing at (x, y), facing the camera."""
    j = {
        "nose": (0, 0, 1.62), "left_eye": (-0.03, 0, 1.66), "right_eye": (0.03, 0, 1.66), "left_ear": (-0.08, 0.02, 1.64),
        "right_ear": (0.08, 0.02, 1.64), "left_shoulder": (-0.19, 0, 1.45), "right_shoulder": (0.19, 0, 1.45),
        "left_elbow": (-0.25, 0, 1.17), "right_elbow": (0.25, 0, 1.17), "left_wrist": (-0.42, 0, 0.9),
        "right_wrist": (0.42, 0.0, 0.9 + wave), "left_hip": (-0.1, 0, 0.95), "right_hip": (0.1, 0, 0.95),
        "left_knee": (-0.11, 0, 0.5), "right_knee": (0.11, 0, 0.5), "left_ankle": (-0.11, 0, 0.08), "right_ankle": (0.11, 0, 0.08),
    }
    return {k: np.array([x + b, y + a, c]) for k, (a, b, c) in j.items()}  # a = left/right (room y), b = toward/away from camera (room x)


def _project(p_room):
    pc = R_INV @ p_room + T_INV
    return INTR.fx * pc[0] / pc[2] + INTR.cx, INTR.fy * pc[1] / pc[2] + INTR.cy, pc[2]


def _scene(x=3.0, y=0.3, surface=0.05, skip_depth=(), wall=()):
    """keypoints (pixel) + depth image whose patches hold the body-surface depth (joint depth - surface)."""
    depth = np.full((480, 640), 0, dtype=np.uint16)
    kps, truth = {}, _body(x, y)
    for name in COCO_KEYPOINT_NAMES:
        u, v, d = _project(truth[name])
        kps[name] = (u, v, 0.9)
        if name in skip_depth:
            continue
        dd = d + 2.5 if name in wall else d - surface
        depth[int(v) - 5:int(v) + 6, int(u) - 5:int(u) + 6] = int(round(dd / INTR.depth_scale))
    return depth, kps, truth


def _lift(depth, kps, **kw):
    cfg = JointsConfig(enabled=True)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return lift_joints(depth, kps, INTR, FT, cfg, ref_depth_m=kw.get("ref"))


def test_lifted_joints_match_truth():
    depth, kps, truth = _scene()
    out = lift_joints(depth, kps, INTR, FT, JointsConfig(enabled=True), ref_depth_m=_project(truth["left_hip"])[2])
    assert out is not None and len(out) == 17
    for name, j in zip(COCO_KEYPOINT_NAMES, out):
        assert j is not None, name
        assert np.linalg.norm(np.array(j[:3]) - truth[name]) < 0.02, (name, j, truth[name])
        assert j[3] == 0.9


def test_missing_and_low_confidence_keypoints_are_null():
    depth, kps, truth = _scene()
    kps["left_wrist"] = (0.0, 0.0, 0.0)  # how M3 zeroes a keypoint it is unsure of
    kps["right_knee"] = (kps["right_knee"][0], kps["right_knee"][1], 0.1)
    out = lift_joints(depth, kps, INTR, FT, JointsConfig(enabled=True))
    assert out[COCO_KEYPOINT_NAMES.index("left_wrist")] is None
    assert out[COCO_KEYPOINT_NAMES.index("right_knee")] is None
    assert sum(j is not None for j in out) == 15


def test_depth_hole_on_thin_limb_falls_back_to_torso_depth():
    depth, kps, truth = _scene(skip_depth=("right_wrist",))
    ref = _project(truth["left_hip"])[2]
    out = lift_joints(depth, kps, INTR, FT, JointsConfig(enabled=True), ref_depth_m=ref)
    j = out[COCO_KEYPOINT_NAMES.index("right_wrist")]
    assert j is not None and j[3] == 0.45  # confidence halved
    # lands on the SAME camera ray as the true wrist, at torso depth (+offset)
    pc = R_INV @ np.array(j[:3]) + T_INV
    u, v, _ = _project(truth["right_wrist"])
    assert abs(INTR.fx * pc[0] / pc[2] + INTR.cx - u) < 0.5 and abs(INTR.fy * pc[1] / pc[2] + INTR.cy - v) < 0.5
    assert abs(pc[2] - (ref + 0.05)) < 0.002


def test_wrist_depth_on_wall_behind_person_is_rejected():
    depth, kps, truth = _scene(wall=("left_wrist", "right_ankle"))
    ref = _project(truth["left_hip"])[2]
    out = lift_joints(depth, kps, INTR, FT, JointsConfig(enabled=True), ref_depth_m=ref)
    for n in ("left_wrist", "right_ankle"):
        j = out[COCO_KEYPOINT_NAMES.index(n)]
        assert np.linalg.norm(np.array(j[:3]) - truth[n]) < 0.7  # NOT 2.5 m behind the person
        assert abs((R_INV @ np.array(j[:3]) + T_INV)[2] - (ref + 0.05)) < 0.002


def test_reference_depth_derived_from_trunk_when_not_given():
    depth, kps, truth = _scene(skip_depth=("left_wrist",))
    out = lift_joints(depth, kps, INTR, FT, JointsConfig(enabled=True))
    assert out is not None and out[COCO_KEYPOINT_NAMES.index("left_wrist")] is not None


def test_too_few_joints_far_person_or_no_depth_gives_none():
    depth, kps, truth = _scene()
    few = {k: kps[k] for k in list(kps)[:4]}
    assert lift_joints(depth, few, INTR, FT, JointsConfig(enabled=True)) is None
    assert lift_joints(depth, kps, INTR, FT, JointsConfig(enabled=True), ref_depth_m=8.0) is None
    assert lift_joints(None, kps, INTR, FT, JointsConfig(enabled=True)) is None
    assert lift_joints(depth, {}, INTR, FT, JointsConfig(enabled=True)) is None
    bad = dict(kps); bad["nose"] = (float("nan"), 5.0, 0.9); bad["left_eye"] = (9999.0, 5.0, 0.9)
    out = lift_joints(depth, bad, INTR, FT, JointsConfig(enabled=True))
    assert out[0] is None and out[1] is None and out[5] is not None


# ---------------------------------------------------------------- through the pipeline and onto the wire

def _frame_with_skeleton(i, x=3.0, y=0.3):
    frame, det = _make_person_detection(FT, x, y, i)
    depth, kps, truth = _scene(x, y)
    # keep the torso block the base helper painted, add the limb patches
    m = depth > 0
    frame.depth[m] = depth[m]
    det.keypoints = kps
    return frame, det, truth


def _run(enabled, n=12, phase="B"):
    cfg = M4Config.default(); cfg.phase = phase; cfg.track.tentative_confirm_hits = 2; cfg.joints.enabled = enabled
    pipe = M4Pipeline(cfg, FixedFrameProvider(FT))
    last, truth = [], None
    for i in range(n):
        frame, det, truth = _frame_with_skeleton(i)
        recs = pipe.process(frame, [det])
        if recs: last = recs
    return pipe, last, truth


def test_pipeline_off_by_default_sends_nothing_new():
    _, recs, _ = _run(False)
    assert recs and recs[0].joints is None
    w = record_to_wire(recs[0])
    assert "joints" not in w.to_dict()


def test_pipeline_on_sends_17_room_frame_joints_close_to_truth_and_matches_schema():
    _, recs, truth = _run(True)
    rec = recs[0]
    assert rec.joints is not None and len(rec.joints) == 17
    errs = [np.linalg.norm(np.array(j[:3]) - truth[n]) for n, j in zip(COCO_KEYPOINT_NAMES, rec.joints) if j is not None]
    assert len(errs) >= 15 and max(errs) < 0.6 and np.median(errs) < 0.1, (max(errs), np.median(errs))
    # standing up: nose above hips above ankles, in the room frame
    z = {n: j[2] for n, j in zip(COCO_KEYPOINT_NAMES, rec.joints) if j}
    assert z["nose"] > z["left_hip"] > z["left_ankle"]
    msg = tracks_message(map_id="m", cam_id="cam0", frame="room", seq=1, t_capture=0.0, t_publish=0.0, tracks=[record_to_wire(rec)])
    schema = json.loads((Path(__file__).resolve().parents[2] / "contracts/schemas/poi_tracks.schema.json").read_text())
    jsonschema.validate(json.loads(json.dumps(msg)), schema)
    assert len(msg["tracks"][0]["joints"]) == 17


def test_phase_a_record_carries_joints_too():
    _, recs, _ = _run(True, phase="A")
    assert recs[0].joints is not None and "joints" in recs[0].to_dict()


def test_coasting_track_drops_stale_pose_after_max_age():
    pipe, recs, _ = _run(True)
    assert recs[0].joints is not None
    t = recs[0].age_s
    seen = []
    for k in range(1, 20):  # nobody detected any more
        frame, _ = _make_person_detection(FT, 3.0, 0.3, 11 + k)
        seen.append(pipe.process(frame, [])[0])
    assert seen[0].state == "coasting" and seen[0].joints is not None  # 1/30 s later: pose still fresh
    assert seen[-1].state == "coasting" and seen[-1].joints is None  # 0.6 s later: pose dropped, track kept


def test_jsonl_roundtrip_keeps_joints(tmp_path):
    from poi_localization.io.world_track import JsonlWorldTrackWriter, read_jsonl
    _, recs, _ = _run(True)
    p = tmp_path / "t.jsonl"
    w = JsonlWorldTrackWriter(p, "B", "m"); w.write_frame(0.0, 0, recs); w.close()
    back = list(read_jsonl(p))[0][4][0]
    assert back.joints is not None and len(back.joints) == 17
    # old logs (no "joints" key) still load
    row = json.loads(p.read_text()); [t.pop("joints") for t in row["tracks"]]
    p2 = tmp_path / "old.jsonl"; p2.write_text(json.dumps(row) + "\n")
    assert list(read_jsonl(p2))[0][4][0].joints is None
