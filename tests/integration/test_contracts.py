"""Every contract is validated at BOTH ends with the same schema (producer test + consumer test)."""
import dataclasses
import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")
from sts.contracts import validate, validate_file

EX = Path(__file__).resolve().parents[2] / "docs" / "schema_examples"
GOLDEN = {"tracks_room_phase_b": "poi_tracks", "tracks_confirmed": "poi_tracks", "tracks_empty": "poi_tracks",
          "event_track_start": "poi_event", "event_track_end": "poi_event", "health_ok": "poi_health",
          "scene_phase_a": "poi_scene"}


@pytest.mark.parametrize("name,schema", sorted(GOLDEN.items()))
def test_golden_wire_examples_validate(name, schema):
    assert validate_file(EX / f"{name}.json", schema) == []


def test_schemas_reject_garbage():
    assert validate({"type": "tracks"}, "poi_tracks")
    assert validate({"type": "health", "fps": 1, "calib": "great", "clients": 0, "clock": "synced"}, "poi_health")
    assert validate({"origin": [0, 0], "resolution": 0, "grid": [[True]]}, "walkable")


# ---- producer side: what M2 actually writes
def test_m2_calibration_validates(synth_master):
    a = synth_master.anchor_dir
    assert validate_file(a / "calibration.cam0.json", "calibration") == []
    assert validate_file(a / "walkable.json", "walkable") == []
    assert validate_file(a / "room_frame.json", "room_frame") == []


# ---- consumer side: M4's own loader reads what M2 wrote
def test_m4_loads_what_m2_wrote(synth_master):
    import numpy as np
    from poi_localization.frames.room_frame import load_room_frame
    from poi_localization.tracking.gates import load_walkable_grid
    a = synth_master.anchor_dir
    tf = load_room_frame(a / "calibration.cam0.json", expected_map_id=synth_master.map_id)
    cal = json.loads((a / "calibration.cam0.json").read_text())
    assert np.allclose(tf.R, np.array(cal["T_room_cam"])[:3, :3])
    assert tf.map_id == synth_master.map_id
    assert 1.5 < tf.camera_height_m < 3.0
    grid = load_walkable_grid(a / "walkable.json")
    assert grid is not None


def test_m4_refuses_a_calibration_from_another_map(synth_master):
    from poi_localization.frames.room_frame import CalibrationMapIdMismatch, load_room_frame
    with pytest.raises(CalibrationMapIdMismatch):
        load_room_frame(synth_master.anchor_dir / "calibration.cam0.json", expected_map_id="000000000000")


def test_m5_scene_consumes_the_anchor_bundle(synth_master):
    from sts.configgen import write_configs
    from poi_present.config import PresentConfig
    from poi_present.server.app import SceneState
    g = write_configs(synth_master.site, "cam0", phase="B", map_id=synth_master.map_id)
    sc = SceneState(PresentConfig.load(g.present_path))
    kinds = {a.kind for a in sc.assets}
    assert "points" in kinds, "M5 must discover anchors/<site>/cam0/viewer/points.ply"
    assert sc.walkable and {"origin", "resolution", "grid"} <= set(sc.walkable)
    assert validate(json.loads(json.dumps(sc.to_doc().to_dict())), "poi_scene") == []


def test_frame_and_intrinsics_types_stay_field_compatible():
    """Duplicated on purpose (M3 must never import pyslam). If either side grows a field, this fails."""
    from pyslam.core.types import Frame as F1, Intrinsics as I1
    from poi_perception.contracts import Frame as F2, Intrinsics as I2
    i1 = {f.name for f in dataclasses.fields(I1)}
    i2 = {f.name for f in dataclasses.fields(I2)}
    assert i2 <= i1, f"poi_perception Intrinsics has fields pyslam's lacks: {i2 - i1}"
    f1 = {f.name for f in dataclasses.fields(F1)}
    f2 = {f.name for f in dataclasses.fields(F2)}
    # M3's Frame is a SUBSET view; the attributes M3/M4/watchdog read must exist on both.
    for needed in ("t", "rgb", "depth", "intr"):
        assert needed in f1 and needed in f2, needed


def test_published_messages_validate(synth_master):
    """The M5 adapter's wire output (the real record_to_wire path) validates against poi.v1."""
    from poi_present.schema import event_message, health_message, tracks_message, TrackWire
    t = TrackWire(id=3, state="confirmed", p=(1, 2, 0), v=(0, 0, 0), cov_xy=(0.01, 0, 0.01), height=1.7, conf=0.9,
                  src="fused", age_s=1.0, bbox=(1, 2, 3, 4))
    m = tracks_message(map_id=synth_master.map_id, cam_id="cam0", frame="room", seq=1, t_capture=1.0, t_publish=1.1, tracks=[t])
    assert validate(m, "poi_tracks") == []
    assert validate(event_message(event="calib_suspect", track_id=-1, t=1.0, detail="x"), "poi_event") == []
    assert validate(health_message(fps=30, calib="suspect", map_id=None, temp_c=None, drop_rate=0, latency_ms=None, clients=0), "poi_health") == []
