"""The runtime watchdog supervisor, driven with depth rendered from the REAL synthetic site.

Truth table being verified:
  live depth == reference                      -> startup verified, state ok
  camera yawed 4 deg (depth layer catches it)  -> suspect, event emitted exactly once, 3D suppressed
  suspect never clears on depth alone          -> only a converged ICP recheck at the calibrated pose clears it
  broken check / wrong-shaped frame            -> suspect (never silently ok)
"""
import types

import numpy as np
import pytest

from sts.calib_health import CalibState
from sts.runtime import build_anchor_watchdog, make_map_loader
from sts.site import WatchdogSpec
from sts.watchdog_runtime import WatchdogSupervisor
from tests.synth.anchor_world import INTR, cam_pose_room, render_capture
from tests.synth.site_fixture import T_TRUE_ROOM_CAM, frames_from_capture_depth


class FakeClock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t
    def advance(self, s): self.t += s


def _intr():
    from poi_perception.contracts import Intrinsics
    return Intrinsics(fx=INTR["fx"], fy=INTR["fy"], cx=INTR["cx"], cy=INTR["cy"], width=INTR["width"], height=INTR["height"],
                      depth_scale=INTR["depth_scale"], baseline=INTR.get("baseline", 0.05))


def _frame(depth_raw, i=0):
    from poi_perception.contracts import Frame
    return Frame(t=float(i) / 30, rgb=np.zeros((480, 640, 3), np.uint8), depth=depth_raw, intr=_intr(), frame_id=i)


@pytest.fixture(scope="module")
def depths(synth_master):
    w = synth_master.world
    true = render_capture(w, T_TRUE_ROOM_CAM, n_frames=12, seed=5).depth_med
    yaw4 = render_capture(w, cam_pose_room(1.0, 3.4, 2.3, -25 + 4.0, 28), n_frames=12, seed=6).depth_med
    yaw15 = render_capture(w, cam_pose_room(1.0, 3.4, 2.3, -25 + 1.5, 28), n_frames=12, seed=7).depth_med
    return {"true": frames_from_capture_depth(true), "yaw4": frames_from_capture_depth(yaw4), "yaw15": frames_from_capture_depth(yaw15)}


def _sup(synth, *, icp=False, clock=None, spec=None, events=None):
    wd, acfg = build_anchor_watchdog(synth.site, "cam0")
    calib = CalibState("missing")
    spec = spec or synth.site.watchdog
    loader = make_map_loader(synth.site, "cam0", acfg) if icp else None
    clock = clock or FakeClock()
    sup = WatchdogSupervisor(spec, calib, wd, anchor_cfg=acfg, map_loader=loader, inline=True, clock=clock,
                             on_event=(lambda k, r: events.append((k, r))) if events is not None else None)
    return sup, calib, clock


def _feed(sup, depth, n, n_tracks=0, start=0, clock=None, dt=0.0):
    for i in range(n):
        sup.observe(_frame(depth, start + i), n_tracks)
        if clock:
            clock.advance(dt)


def test_starts_suspect_until_the_first_check_then_ok(synth, depths):
    sup, calib, clock = _sup(synth)
    assert calib.state() == "suspect" and "startup" in calib.reason()      # nothing is published as trusted before verification
    _feed(sup, depths["true"], 12)
    assert calib.state() == "ok", calib.reason()
    assert sup.should_publish_3d()


def test_missing_calibration_reports_missing(synth):
    calib = CalibState("ok")
    WatchdogSupervisor(synth.site.watchdog, calib, None)
    assert calib.state() == "missing"


def test_moved_camera_is_suspect_event_once_and_3d_suppressed(synth, depths):
    events = []
    sup, calib, clock = _sup(synth, events=events)
    _feed(sup, depths["true"], 12)
    assert calib.state() == "ok"
    # 4 deg yaw: the depth layer needs `wd_consecutive` consecutive breaches; the confirm cadence makes that seconds, not minutes
    for k in range(6):
        clock.advance(10.0)
        _feed(sup, depths["yaw4"], 12, start=100 * (k + 1))
        if calib.state() == "suspect":
            break
    assert calib.state() == "suspect", "a 4 degree yaw must be caught by the depth layer"
    assert [e[0] for e in events] == ["calib_suspect"], events
    assert not sup.should_publish_3d()
    # depth alone must never clear it, even if the scene looks right again
    clock.advance(10.0)
    _feed(sup, depths["true"], 12, start=900)
    assert calib.state() == "suspect"


def test_flag_policy_keeps_publishing_but_still_reports_suspect(synth, depths):
    spec = WatchdogSpec(**{**synth.site.watchdog.__dict__, "on_suspect": "flag"})
    sup, calib, _ = _sup(synth, spec=spec)
    _feed(sup, depths["yaw4"], 12)
    assert calib.state() == "suspect" and sup.should_publish_3d()


def test_wrong_shaped_stream_is_suspect(synth, depths):
    sup, calib, _ = _sup(synth)
    _feed(sup, depths["true"][::2, ::2].copy(), 12)
    assert calib.state() == "suspect" and "shape" in calib.reason()


def test_a_person_occluding_part_of_the_image_does_not_trip_it(synth, depths):
    d = depths["true"].copy()
    d[150:400, 250:330] = 1500                     # ~7% of the image, a person-sized occluder
    sup, calib, clock = _sup(synth)
    _feed(sup, d, 12, n_tracks=1)
    assert calib.state() == "ok", calib.reason()


def test_a_crashing_check_reads_as_suspect_not_ok(synth, depths, monkeypatch):
    sup, calib, _ = _sup(synth)
    _feed(sup, depths["true"], 12)
    assert calib.state() == "ok"

    def boom(_):
        raise RuntimeError("boom")
    monkeypatch.setattr(sup.wd, "check_depth", boom)
    sup._next_depth = 0
    _feed(sup, depths["true"], 12, start=50)
    assert calib.state() == "suspect" and "boom" in calib.reason() and sup.stats["errors"] == 1


def test_observe_is_cheap_and_never_raises_without_a_watchdog(depths):
    calib = CalibState("ok")
    sup = WatchdogSupervisor(WatchdogSpec(), calib, None)
    sup.observe(_frame(depths["true"]), 0)


@pytest.mark.slow
def test_icp_recheck_detects_a_small_yaw_the_depth_layer_misses_and_clears_only_when_back(synth, depths):
    """1.5 deg yaw is below the depth layer's trigger (documented); the ICP layer must catch it, and only an
    'ok' ICP verdict at the calibrated pose may clear it."""
    sup, calib, clock = _sup(synth, icp=True)
    _feed(sup, depths["true"], 40)                        # startup verification + first idle ICP recheck
    assert calib.state() == "ok", calib.reason()
    assert sup.last_recheck is not None and sup.last_recheck["verdict"] == "ok", sup.last_recheck
    # now the camera is nudged by 1.5 deg yaw
    for k in range(6):
        clock.advance(5.0)
        _feed(sup, depths["yaw15"], 40, start=1000 * (k + 1))
        if calib.state() == "suspect":
            break
    assert calib.state() == "suspect", f"ICP recheck must catch 1.5 deg; last={sup.last_recheck}"
    assert sup.last_recheck["verdict"] == "moved" and sup.last_recheck["shift_deg"] > 0.3
    # back at the calibrated pose: the recheck clears it
    clock.advance(120.0)
    _feed(sup, depths["true"], 40, start=50000)
    assert calib.state() == "ok", (calib.reason(), sup.last_recheck)
