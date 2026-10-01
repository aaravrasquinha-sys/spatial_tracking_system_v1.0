"""
_build_post_processing_filters is pure aside from the `rs` module it's
handed -- no real librealsense/camera needed to test its logic (filter
ordering, config values reaching the right options, hole-filling
toggling on/off). A lightweight fake `rs` module stands in for
pyrealsense2, which isn't installed in this (or most CI) environments.
"""
from types import SimpleNamespace

import pytest

from poi_perception.capture.realsense_source import _build_post_processing_filters
from poi_perception.config import CameraConfig


class _FakeFilter:
    def __init__(self, kind):
        self.kind = kind
        self.options = {}

    def set_option(self, option, value):
        self.options[option] = value


class _FakeDisparityTransform:
    def __init__(self, to_disparity):
        self.to_disparity = to_disparity


def _fake_rs():
    return SimpleNamespace(
        disparity_transform=lambda to_disparity: _FakeDisparityTransform(to_disparity),
        spatial_filter=lambda: _FakeFilter("spatial"),
        temporal_filter=lambda: _FakeFilter("temporal"),
        hole_filling_filter=lambda: _FakeFilter("hole_filling"),
        option=SimpleNamespace(
            filter_magnitude="filter_magnitude",
            filter_smooth_alpha="filter_smooth_alpha",
            filter_smooth_delta="filter_smooth_delta",
            holes_fill="holes_fill",
        ),
    )


def test_filter_chain_ordering_brackets_spatial_temporal_with_disparity_transform():
    cfg = CameraConfig()
    rs = _fake_rs()
    filters = _build_post_processing_filters(cfg, rs)
    kinds = [getattr(f, "kind", "disparity_transform") for f in filters]
    assert kinds[0] == "disparity_transform" and filters[0].to_disparity is True
    assert kinds[1] == "spatial"
    assert kinds[2] == "temporal"
    assert kinds[3] == "disparity_transform" and filters[3].to_disparity is False
    assert kinds[4] == "hole_filling"


def test_hole_filling_omitted_when_disabled():
    cfg = CameraConfig(hole_filling_mode=0)
    rs = _fake_rs()
    filters = _build_post_processing_filters(cfg, rs)
    kinds = [getattr(f, "kind", "disparity_transform") for f in filters]
    assert "hole_filling" not in kinds
    assert len(filters) == 4


def test_config_values_reach_the_filter_options():
    cfg = CameraConfig(spatial_filter_magnitude=3.0, temporal_filter_smooth_alpha=0.2)
    rs = _fake_rs()
    filters = _build_post_processing_filters(cfg, rs)
    spatial = next(f for f in filters if getattr(f, "kind", None) == "spatial")
    temporal = next(f for f in filters if getattr(f, "kind", None) == "temporal")
    assert spatial.options["filter_magnitude"] == 3.0
    assert temporal.options["filter_smooth_alpha"] == 0.2


def test_camera_config_round_trips_new_tuning_fields():
    cfg = CameraConfig.__class__ if False else CameraConfig  # keep import graph simple
    c = cfg(laser_power=150.0, visual_preset="high_density", enable_post_processing=False)
    assert c.laser_power == 150.0
    assert c.visual_preset == "high_density"
    assert c.enable_post_processing is False
