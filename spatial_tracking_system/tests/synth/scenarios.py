"""
WP-A2 fixture set (Phase 1 plan section 4/6, gate G1A.3).

Every fixture here is checked by validate_scenario() before being
returned -- see build() at the bottom, which raises rather than silently
handing back a scenario that violates the free-space/depth/smoothness
bounds. This replaces run_synth.py's square_loop_scenario/
corridor_scenario, which had two Phase-0 defects this module exists to
fix (see the Phase 1 plan, section 0 and findings F3/F4):

  1. square_loop_scenario's path ended with a single-frame heading snap
     (poses_from_path's look-ahead clamps instead of wrapping at a path
     end -- fixed generally in world.py's poses_from_path(loop=...)).
  2. corridor_scenario's four box segments didn't overlap, so roughly
     half the path was outside all of them, looking into empty space or
     through walls -- replaced with a proper overlapping ring geometry
     (see add_ring_corridor below), each corner sharing a real 2x2m
     overlap so the corridor is one connected free-space region.

Every fixture is 6-DoF (tests/synth/world.py::handheld_poses), unlike
Phase 0's flat, yaw-only paths, which cannot distinguish a correct
odometry axis convention from an inverted one (Phase 1 plan finding F2).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from pyslam.core.types import Intrinsics
from pyslam.core import lie
from tests.synth.world import (
    World, spline_trajectory, poses_from_path, handheld_poses, validate_scenario, T_BODY_CAM
)

RIG_INTRINSICS = Intrinsics(fx=606.75, fy=606.57, cx=320.19, cy=237.06,
                             width=640, height=480,
                             depth_scale=0.0010000000474974513, baseline=0.0499)


@dataclass
class Scenario:
    name: str
    world: World
    poses: List[np.ndarray]
    dt: float
    free_space_boxes: List[Tuple[np.ndarray, np.ndarray]]
    intr: Intrinsics = RIG_INTRINSICS
    session_breaks: Tuple[int, ...] = ()   # frame indices where a new SLAM session should start
                                            # (used by aliasing_rooms: a physical teleport, not a real path)
    validator_kwargs: dict = None

    def __post_init__(self):
        if self.validator_kwargs is None:
            self.validator_kwargs = {}


# ---------------------------------------------------------------- fixtures

def square6dof(seed: int = 1) -> Scenario:
    """The flagship loop, fixed: same footprint as Phase 0's
    square_loop_scenario, but 6-DoF handheld motion and a wrapped
    look-ahead so there is no end-of-path heading snap."""
    w = World()
    w.add_box_room(np.array([0, 0, 0]), np.array([3, 3, 2.4]), seed0=1)
    waypoints = np.array([
        [0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [0.7, 0.7, 0.0], [0.0, 0.7, 0.0],
    ])
    path = spline_trajectory(waypoints, n_samples=140, loop=True)
    poses = handheld_poses(path, dt=1 / 10.0, seed=seed, roll_deg=8, pitch_deg=6,
                            bob_m=0.03, jitter_deg=2.5, look_ahead=5, loop=True)
    free = [(np.array([0, 0, 0]), np.array([2.6, 2.6, 2.0]))]  # 0.2m clearance margin inside the 3x3 room
    return Scenario("square6dof", w, poses, dt=1 / 10.0, free_space_boxes=free)


def room_orbit(seed: int = 2) -> Scenario:
    """A look-around orbit near the room centre, with continuous yaw
    rotation bursts up to ~100deg/s -- deliberately high angular rate
    (this IS the point of the fixture, not a defect), tests tracking
    through fast rotation rather than translation."""
    w = World()
    w.add_box_room(np.array([0, 0, 0]), np.array([4, 4, 2.4]), seed0=11)
    n = 200
    dt = 1 / 10.0
    poses = []
    radius = 0.4
    for i in range(n):
        t = i / n
        pos = np.array([radius * np.cos(2 * np.pi * t), radius * np.sin(2 * np.pi * t), 0.0])
        yaw = 2 * np.pi * 2.2 * t + 0.6 * np.sin(2 * np.pi * 0.7 * t)
        pitch = np.radians(6.0) * np.sin(2 * np.pi * 0.5 * t + 0.3)
        roll = np.radians(5.0) * np.sin(2 * np.pi * 0.6 * t)
        R = lie.so3_exp(np.array([0, 0, yaw])) @ lie.so3_exp(np.array([roll, pitch, 0]))
        pos[2] += 0.02 * np.sin(2 * np.pi * 0.9 * t)
        poses.append(lie.make_T(R, pos))
    free = [(np.array([0, 0, 0]), np.array([3.4, 3.4, 2.0]))]
    return Scenario("room_orbit", w, poses, dt=dt, free_space_boxes=free,
                     validator_kwargs=dict(max_ang_vel_deg_s=250.0))  # spin is intentional here


def add_ring_corridor(w: World, seed0: int = 100) -> List[Tuple[np.ndarray, np.ndarray]]:
    """A proper hollow rectangular corridor: four 2m-wide hallway
    segments forming a ring, each pair overlapping at a 2x2m corner
    square so the ring is one connected free-space region (see this
    module's docstring for why Phase 0's version wasn't)."""
    boxes = [
        (np.array([4.0, 1.0, 0.0]), np.array([8.0, 2.0, 2.4])),   # bottom
        (np.array([7.0, 4.0, 0.0]), np.array([2.0, 8.0, 2.4])),   # right
        (np.array([4.0, 7.0, 0.0]), np.array([8.0, 2.0, 2.4])),   # top
        (np.array([1.0, 4.0, 0.0]), np.array([2.0, 8.0, 2.4])),   # left
    ]
    for i, (c, s) in enumerate(boxes):
        w.add_box_room(c, s, seed0=seed0 + i * 10)
    return boxes


def corridor_v2(seed: int = 3) -> Scenario:
    """The corrected corridor loop (Phase 1 plan finding F4): a real
    ~24m connected ring, centreline path with ~1m clearance from every
    wall, validated frame by frame rather than assumed."""
    w = World()
    boxes = add_ring_corridor(w)
    waypoints = np.array([
        [1, 1, 0], [3, 1, 0], [5, 1, 0], [6.5, 1, 0],
        [7, 1.5, 0], [7, 3, 0], [7, 5, 0], [7, 6.5, 0],
        [6.5, 7, 0], [5, 7, 0], [3, 7, 0], [1.5, 7, 0],
        [1, 6.5, 0], [1, 5, 0], [1, 3, 0], [1, 1.5, 0],
    ])
    path = spline_trajectory(waypoints, n_samples=480, loop=True)
    poses = handheld_poses(path, dt=1 / 10.0, seed=seed, roll_deg=7, pitch_deg=5,
                            bob_m=0.03, jitter_deg=2.0, look_ahead=6, loop=True)
    return Scenario("corridor_v2", w, poses, dt=1 / 10.0, free_space_boxes=boxes,
                     validator_kwargs=dict(min_clearance_m=0.35, max_frame_rot_ratio=3.5))


def corridor_lap13(seed: int = 7) -> Scenario:
    """WP-L: corridor_v2's ring driven for 1.3 laps. corridor_v2 itself is ONE lap
    that ends exactly where it started, so only a single keyframe has a true
    revisit partner (measured, WP_L_Findings.md) and no loop-closure change can be
    evaluated on it. Here the last 30% of the run re-traverses the first 30% of the
    ring in the SAME direction (the easy case for appearance retrieval, deliberately:
    this fixture asks 'is the system able to close a real ~24 m loop at all, and does
    the start of the run survive in memory long enough', not 'is retrieval robust
    to viewpoint change')."""
    w = World()
    boxes = add_ring_corridor(w)
    waypoints = np.array([
        [1, 1, 0], [3, 1, 0], [5, 1, 0], [6.5, 1, 0],
        [7, 1.5, 0], [7, 3, 0], [7, 5, 0], [7, 6.5, 0],
        [6.5, 7, 0], [5, 7, 0], [3, 7, 0], [1.5, 7, 0],
        [1, 6.5, 0], [1, 5, 0], [1, 3, 0], [1, 1.5, 0],
    ])
    lap = spline_trajectory(waypoints, n_samples=480, loop=True)
    n_keep = 624                                        # 1.3 laps
    path = np.concatenate([lap, lap[:n_keep - 480 + 20]], axis=0)   # +20 samples of look-ahead margin
    poses = handheld_poses(path, dt=1 / 10.0, seed=seed, roll_deg=7, pitch_deg=5,
                            bob_m=0.03, jitter_deg=2.0, look_ahead=6, loop=False)[:n_keep]
        # the margin frames are generated then dropped: with loop=False the heading
        # look-ahead clamps at the end of the path and the last few poses snap in yaw
        # (validate_scenario caught exactly that on the first attempt)
    return Scenario("corridor_lap13", w, poses, dt=1 / 10.0, free_space_boxes=boxes,
                     validator_kwargs=dict(min_clearance_m=0.35, max_frame_rot_ratio=3.5))


def static_60s(seed: int = 4) -> Scenario:
    """Camera stationary on a desk (see Runbook.md's own desk_static.bag
    protocol) -- the drift/noise-floor calibration fixture. Small
    handheld tremor only, no deliberate motion."""
    w = World()
    w.add_box_room(np.array([0, 0, 0]), np.array([4, 4, 2.4]), seed0=21)
    n = 600  # 60s @ 10Hz
    dt = 0.1
    path = np.tile(np.array([0.0, 0.0, 0.0]), (n, 1))
    poses = handheld_poses(path, dt=dt, seed=seed, roll_deg=0.5, roll_hz=0.3,
                            pitch_deg=0.5, pitch_hz=0.25, bob_m=0.005, jitter_deg=0.3,
                            look_ahead=1)
    free = [(np.array([0, 0, 0]), np.array([3.4, 3.4, 2.0]))]
    return Scenario("static_60s", w, poses, dt=dt, free_space_boxes=free)


def aliasing_rooms(seed: int = 5) -> Scenario:
    """Two rooms 20m apart with IDENTICAL texture seeds (deliberate
    perceptual aliasing -- world.py's add_two_rooms). The pipeline is
    fed a short loop in room A, then a short loop in room B, with NO
    physically-connecting path between them (an instantaneous cut, not
    a rendered transit through the 20m of empty space between the two
    rooms, which the camera can't usefully see anyway). session_breaks
    marks the cut frame index so the runner can treat it as a new
    tracking session (matching the WP-B3 session-on-LOST design) rather
    than expecting continuous odometry across a jump that never
    happened physically."""
    w = World()
    w.add_two_rooms(seed_a=200, seed_b=200)
    waypoints = np.array([[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0]])
    path_a = spline_trajectory(waypoints, n_samples=40, loop=True)
    poses_a = handheld_poses(path_a, dt=0.1, seed=seed, roll_deg=6, pitch_deg=5,
                              bob_m=0.02, jitter_deg=1.5, look_ahead=4, loop=True)
    path_b = spline_trajectory(waypoints + np.array([20, 0, 0]), n_samples=40, loop=True)
    poses_b = handheld_poses(path_b, dt=0.1, seed=seed + 1, roll_deg=6, pitch_deg=5,
                              bob_m=0.02, jitter_deg=1.5, look_ahead=4, loop=True)
    poses = poses_a + poses_b
    free = [
        (np.array([0, 0, 0]), np.array([3.4, 3.4, 2.0])),
        (np.array([20, 0, 0]), np.array([3.4, 3.4, 2.0])),
    ]
    return Scenario("aliasing_rooms", w, poses, dt=0.1, free_space_boxes=free,
                     session_breaks=(len(poses_a),),
                     validator_kwargs=dict(max_ang_vel_deg_s=200.0))  # tight 0.5m loops -> naturally fast heading changes


def gravity_init_square6dof(seed: int = 6) -> Scenario:
    """WP-P5.2's own fixture gap, closed: NONE of the other 5 fixtures
    can ever trigger a gravity/tilt prior through a real pipeline run
    (see WP_P5_Findings.md section 4.2) -- the 4 moving ones have
    continuous 6-DoF motion from frame 0 by design (deliberately, to
    avoid the original Phase 0 F2 bug), and `static_60s` never crosses
    the keyframe-creation motion threshold at all (only ever produces
    ONE keyframe, so there is no since-last-keyframe WINDOW to ever
    check). This fixture is a different shape, not a relaxation of
    either constraint:

      Phase A (15 frames, ~1.5s): a genuine static hold, same small
      handheld-tremor levels as `static_60s` itself (see that
      function). Establishes the gravity-aligned reference direction.

      Phase B (30 frames, ~3s): a SMOOTH, CONSTANT-VELOCITY, NON-
      ROTATING straight-line translation toward (0,0,0), approaching
      along the SAME direction `square6dof`'s own loop path actually
      starts heading (measured directly from the loop's own first
      pose, not assumed -- see the code below for why "toward the
      second waypoint" is the wrong assumption for a closed loop).
      This is the key idea: gravity's quasi-static assumption is about
      LOW GYRO RATE and LOW ACCELEROMETER VARIANCE (`gravity.py::
      is_quasi_static`'s own actual criteria), not "zero velocity" --
      a smooth constant-velocity translation has near-zero angular
      rate and near-zero acceleration variance while still
      accumulating real net displacement, comfortably crossing
      `cfg.keyframe_trans_m` and producing several genuine keyframes.
      This is also physically honest: real VIO systems commonly treat
      calm/smooth segments this way, not literally "camera not
      moving at all".

      Phase C: `square6dof`'s own already-validated 6-DoF loop,
      UNCHANGED.

    Reuses square6dof's own room geometry and free-space box exactly
    (world seed0=1) -- Phase A/B's positions ((-0.5,0,0) to (0,0,0))
    sit comfortably inside it (clearance ~0.8m from the nearest wall,
    well over min_clearance_m).
    """
    w = World()
    w.add_box_room(np.array([0, 0, 0]), np.array([3, 3, 2.4]), seed0=1)
    dt = 1 / 10.0

    waypoints = np.array([
        [0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [0.7, 0.7, 0.0], [0.0, 0.7, 0.0],
    ])
    path = spline_trajectory(waypoints, n_samples=140, loop=True)
    main_poses = handheld_poses(path, dt=dt, seed=seed, roll_deg=8, pitch_deg=6,
                                 bob_m=0.03, jitter_deg=2.5, look_ahead=5, loop=True)

    # The loop's own starting heading is a BLENDED tangent (Catmull-Rom
    # wraps the last waypoint's incoming direction into the first
    # point's tangent for a closed loop), NOT simply "toward the second
    # waypoint" -- an initial version of this fixture approached along
    # hardcoded +x and failed validate_scenario with a 30deg snap at
    # this exact boundary. Fixed by measuring the loop's actual starting
    # heading and approaching along THAT direction instead.
    initial_heading = main_poses[0][:3, 0].copy()
    initial_heading /= np.linalg.norm(initial_heading)

    static_path = -0.5 * initial_heading + np.linspace(-0.01, 0.0, 16)[:, None] * initial_heading
    # A perfectly repeated position (identical waypoints) makes
    # poses_from_path's own forward-vector computation degenerate
    # (norm~0 -> falls back to a hardcoded world +x, REGARDLESS of
    # initial_heading -- caught by validate_scenario itself the first
    # time this used np.tile(...) unchanged: a 30deg snap appeared at
    # THIS boundary instead, the exact same +x-vs-initial_heading
    # mismatch relocated, not a new bug). A 1cm creep along
    # initial_heading over the whole static phase keeps a well-defined
    # heading throughout while staying far below any keyframe or
    # quasi-static threshold. 16 points generated, only the first 15
    # kept -- poses_from_path's own look_ahead clamps (degenerates) at
    # a loop=False path's LAST sample by construction (its own
    # docstring's documented F3-class behaviour); this run into that
    # exact trap a second time before being fixed the same way F3
    # itself was: don't let the sample that needs a valid look-ahead
    # BE the last sample in the array.
    static_poses = handheld_poses(static_path, dt=dt, seed=seed, roll_deg=0.5, roll_hz=0.3,
                                   pitch_deg=0.5, pitch_hz=0.25, bob_m=0.005, jitter_deg=0.3,
                                   look_ahead=1)[:15]

    translate_wp = np.array([-0.5 * initial_heading, [0.0, 0.0, 0.0]])
    translate_path = spline_trajectory(translate_wp, n_samples=31, loop=False)
    # Same look_ahead-clamps-at-the-last-sample trap as the static
    # phase above -- generate one extra sample, keep only the first 30.
    translate_poses = handheld_poses(translate_path, dt=dt, seed=seed + 1, roll_deg=0.5, roll_hz=0.3,
                                      pitch_deg=0.5, pitch_hz=0.25, bob_m=0.0, jitter_deg=0.3,
                                      look_ahead=3)[:30]

    poses = static_poses + translate_poses + main_poses
    free = [(np.array([0, 0, 0]), np.array([2.6, 2.6, 2.0]))]
    return Scenario("gravity_init_square6dof", w, poses, dt=dt, free_space_boxes=free,
                     validator_kwargs=dict(min_clearance_m=0.3))


ALL_SCENARIO_BUILDERS = {
    "square6dof": square6dof,
    "room_orbit": room_orbit,
    "corridor_v2": corridor_v2,
    "corridor_lap13": corridor_lap13,
    "static_60s": static_60s,
    "aliasing_rooms": aliasing_rooms,
    "gravity_init_square6dof": gravity_init_square6dof,
}


def build(name: str, seed: int = 1, validate: bool = True) -> Scenario:
    scen = ALL_SCENARIO_BUILDERS[name](seed=seed)
    if validate:
        ok, report = validate_scenario(scen.world, scen.poses, scen.intr, scen.dt,
                                        scen.free_space_boxes, session_breaks=scen.session_breaks,
                                        **scen.validator_kwargs)
        if not ok:
            raise RuntimeError(f"Scenario '{name}' (seed={seed}) FAILED validate_scenario: {report}")
    return scen


def validate_all(seeds=(1, 2, 3, 4, 5)) -> dict:
    """Run every fixture x every seed through the validator. Used by
    gate G1A.3 and by `python -m tests.synth.scenarios`."""
    results = {}
    for name in ALL_SCENARIO_BUILDERS:
        results[name] = []
        for seed in seeds:
            scen = ALL_SCENARIO_BUILDERS[name](seed=seed)
            ok, report = validate_scenario(scen.world, scen.poses, scen.intr, scen.dt,
                                            scen.free_space_boxes, session_breaks=scen.session_breaks,
                                            **scen.validator_kwargs)
            results[name].append({"seed": seed, "ok": ok, "report": report})
    return results


if __name__ == "__main__":
    results = validate_all()
    all_ok = True
    for name, runs in results.items():
        oks = [r["ok"] for r in runs]
        print(f"{name}: {sum(oks)}/{len(oks)} seeds valid")
        for r in runs:
            if not r["ok"]:
                all_ok = False
                print(f"  seed={r['seed']} FAILED: {r['report']}")
    print("ALL SCENARIOS VALID" if all_ok else "SOME SCENARIOS INVALID -- do not use for gates")
