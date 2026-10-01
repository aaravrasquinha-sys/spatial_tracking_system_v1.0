import json

import pytest

from sts.camera_model import CameraModel, save_camera_model
from sts.configgen import write_configs
from sts.site import SiteConfig, SiteError


def test_defaults_and_roundtrip_preserve_unknown_keys(tmp_path):
    s = SiteConfig.from_dict({"data_dir": str(tmp_path), "future_section": {"a": 1}, "present": {"port": 9000, "new_field": 5}})
    assert s.present.port == 9000 and s.localization.phase == "B" and s.watchdog.on_suspect == "suppress"
    p = tmp_path / "site.json"
    s.save(p)
    assert json.loads(p.read_text())["future_section"] == {"a": 1}
    assert SiteConfig.load(p).present.port == 9000


def test_old_minimal_site_file_still_loads(tmp_path):
    (tmp_path / "s.json").write_text('{"site": "x"}')
    s = SiteConfig.load(tmp_path / "s.json")
    assert s.cameras[0].cam_id == "cam0" and s.retention.track_log_days == 14


@pytest.mark.parametrize("bad", [{"site": "a b"}, {"localization": {"phase": "C"}}, {"watchdog": {"on_suspect": "panic"}},
                                 {"watchdog": {"tilt_layer": "auto"}}, {"perception": {"backend": "onnx"}},
                                 {"cameras": [{"cam_id": "c"}, {"cam_id": "c"}]}])
def test_validation_rejects(bad):
    with pytest.raises(SiteError):
        SiteConfig.from_dict(bad)


def test_camera_selection_and_multicamera_paths(tmp_path):
    s = SiteConfig.from_dict({"data_dir": str(tmp_path), "cameras": [{"cam_id": "cam0"}, {"cam_id": "cam1"}]})
    with pytest.raises(SiteError):
        s.camera()                       # two enabled -> must choose
    assert s.camera("cam1").cam_id == "cam1"
    assert s.anchor_dir("cam0") != s.anchor_dir("cam1")
    assert s.calibration_path("cam1").name == "calibration.cam1.json"
    s2 = SiteConfig.from_dict({"cameras": [{"cam_id": "cam0"}, {"cam_id": "cam1", "enabled": False}]})
    assert s2.camera().cam_id == "cam0"


def test_generated_configs_load_through_module_loaders(tmp_path):
    from poi_localization.config import M4Config
    from poi_perception.config import M3Config
    from poi_present.config import PresentConfig
    s = SiteConfig.from_dict({"data_dir": str(tmp_path), "present": {"port": 9123, "token": "abc"},
                              "cameras": [{"cam_id": "cam0", "width": 640, "height": 480, "serial": "123"}]})
    g = write_configs(s, "cam0", phase="B", map_id="abcdef123456")
    m3, m4, pr = M3Config.load(g.m3_path), M4Config.load(g.m4_path), PresentConfig.load(g.present_path)
    assert (m3.model.input_width, m3.model.input_height) == (640, 480), "engine input must follow the camera, not the 640x640 dataclass default"
    assert m3.camera.serial == "123" and m3.model.backend == "trt"
    assert m4.phase == "B" and m4.phase_b.expected_map_id == "abcdef123456"
    assert pr.server.port == 9123 and pr.server.token == "abc" and pr.scene.frame == "room" and pr.scene.map_id == "abcdef123456"
    assert pr.scene.map_bundle_dir == str(s.anchor_dir("cam0")), "M5 must point at the ANCHOR bundle"
    assert str(tmp_path) in m3.output.jsonl_dir and str(tmp_path) in m4.output.jsonl_dir


def test_phase_a_needs_no_calibration_paths(tmp_path):
    from poi_localization.config import M4Config
    from poi_present.config import PresentConfig
    s = SiteConfig.from_dict({"data_dir": str(tmp_path)})
    g = write_configs(s, "cam0", phase="A")
    assert M4Config.load(g.m4_path).phase_b.calibration_path is None
    assert PresentConfig.load(g.present_path).scene.frame == "local"


def test_overrides_are_deep_merged_and_typos_fail_loudly(tmp_path):
    from poi_localization.config import M4Config
    s = SiteConfig.from_dict({"data_dir": str(tmp_path), "localization": {"overrides": {"track": {"coasting_budget_s": 2.5}}}})
    m4 = M4Config.load(write_configs(s, "cam0").m4_path)
    assert m4.track.coasting_budget_s == 2.5 and m4.track.lost_grace_s == 1.0     # sibling default survives
    bad = SiteConfig.from_dict({"data_dir": str(tmp_path), "localization": {"overrides": {"track": {"coastng_budget_s": 2}}}})
    with pytest.raises(SiteError, match="M4"):
        write_configs(bad, "cam0")


def test_camera_model_is_the_single_source_for_m2_and_m4(tmp_path):
    from poi_localization.config import M4Config
    s = SiteConfig.from_dict({"data_dir": str(tmp_path), "anchor": {"set": {"gate_sigma_trans_m": 0.05}}})
    # null model == no behaviour change
    g0 = write_configs(s, "cam0")
    assert M4Config.load(g0.m4_path).depth_measurement.sigma_disparity_px == 0.1 and g0.anchor_overrides == {"gate_sigma_trans_m": 0.05}
    import sts.site as ss
    save_camera_model(CameraModel(sigma_disparity_px=0.25, depth_bias_frac=0.012, anchor_noise_multiplier=1.5), s.camera_model_path("cam0"))
    try:
        g1 = write_configs(s, "cam0")
        m4 = M4Config.load(g1.m4_path)
        assert m4.depth_measurement.sigma_disparity_px == 0.25 and m4.depth_measurement.depth_bias_frac == 0.012
        assert g1.anchor_overrides == {"sigma_disparity_px": 0.25, "noise_multiplier": 1.5, "gate_sigma_trans_m": 0.05}
        from pyslam.anchor.config import AnchorConfig, apply_overrides
        cfg = apply_overrides(AnchorConfig(), [f"{k}={v}" for k, v in g1.anchor_overrides.items()])
        assert cfg.sigma_disparity_px == 0.25 and cfg.noise_multiplier == 1.5
    finally:
        s.camera_model_path("cam0").unlink()
