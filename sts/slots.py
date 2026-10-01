"""The slot table: every replaceable piece of the system, its contract, and its default.

A slot is a named seam. `site.json -> "slots": {"<slot>": "<implementation>"}` picks
the implementation; omitted slots use the default below. To improve a module you
either (a) edit inside it, keeping its contract, or (b) register a NEW
implementation under a new name and select it per site -- the old one stays
available, and `sts replay --compare` lets you judge the two on recorded data.

This table is also the authoritative list that docs/OPEN_ITEMS.md is checked
against in tests/integration/test_slots.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class SlotSpec:
    name: str
    default: str
    kind: str               # "offline" | "runtime"
    contract: str           # what an implementation must provide / honour
    reserved: tuple = ()    # implementation names documented as planned but NOT built


SLOTS: Dict[str, SlotSpec] = {s.name: s for s in [
    SlotSpec("map_builder", "pyslam_live", "offline",
             "build(site, cam, extra_args) -> Command. Must leave maps/<site>/dense/points.ply (+ capture_profile.json) "
             "in M1's W_cam0 frame. See contracts/map_bundle.md.",
             reserved=("rtabmap_reference", "tsdf_rebuild_from_bag")),
    SlotSpec("map_finalizer", "manifest_lock", "offline",
             "finalize(site, read_only) -> {map_id, manifest_path}. MUST compute map_id over the SAME content files "
             "as M2's resolve_map_id (dense/points.ply + capture_profile.json) or every existing anchor is invalidated. "
             "See contracts/map_bundle.md#map_id-rule.",
             reserved=("acceptance_suite",)),
    SlotSpec("anchor_solver", "pyslam_anchor", "offline",
             "commands(site, cam, ...) -> Command. Must produce the anchor bundle in contracts/anchor_bundle.md; a "
             "rejected solve must be written as calibration.<cam>.REJECTED.json.",
             reserved=("fpfh_cross_check",)),
    SlotSpec("anchor_capture_filter", "none", "offline",
             "mask(frame) -> bool image of pixels to EXCLUDE from the static capture (people). 'none' = empty room required.",
             reserved=("m3_person_mask",)),
    SlotSpec("viewer_asset_producer", "points_only", "offline",
             "produce(site) -> list of files under <anchor>/viewer/ in the ROOM frame. M5 discovers room.glb / points.ply by name.",
             reserved=("room_glb",)),
    SlotSpec("capture_source", "m3_realsense", "runtime",
             "kind(site, cam) -> 'realsense' | 'synthetic' (+ optional playback bag). Frames: poi_perception.contracts.Frame.",
             reserved=("profile_enforcing_realsense", "ir_stream")),
    SlotSpec("detector_backend", "m3_default", "runtime",
             "build(m3_cfg) -> object with the PoseEstimator backend interface (poi_perception.inference.pose_infer).",
             reserved=("int8_experiment",)),
    SlotSpec("frame_provider", "m4_default", "runtime",
             "build(m4_cfg, cam_cfg) -> FrameProvider (poi_localization.frames.frame_provider).",
             reserved=("multi_camera", "online_reanchor")),
    SlotSpec("watchdog_layers", "depth+icp", "runtime",
             "a '+'-joined list of layer names from {depth, icp, tilt}. Each layer reports ok/suspect; any suspect wins.",
             reserved=()),
    SlotSpec("calib_state_provider", "watchdog", "runtime",
             "object with .state() -> 'ok'|'suspect'|'missing' and .reason() -> str; injected into PresentServer.",
             reserved=()),
    SlotSpec("track_source", "live", "runtime",
             "poi_present.sources.base.TrackSource implementation.",
             reserved=("multi_camera_fusion",)),
    SlotSpec("sensor_model", "camera_model_file", "runtime",
             "load(site, cam) -> sts.camera_model.CameraModel; pushed into BOTH M2 and M4 generated configs.",
             reserved=("fitted_from_flat_wall",)),
]}


def slot_names():
    return list(SLOTS)
