# WP-LIVE: System Summary (Modules 1+2 real-time rearchitecture)

**Read this first if you're picking this project up in a new session.**
This document describes ONE work package (call it WP-LIVE) layered on
top of the existing pySLAM repo described in `SYSTEM_SUMMARY.md`,
`README_ORIN.md`, and every `WP_*_Findings.md` file, all of which are
unchanged in spirit and still the "why" for everything WP-LIVE builds
on. This file is WP-LIVE's own "what" and "what's next," in the same
style.

---

## 0. Scope, as approved

Modules 1 (mapping) and 2 (calibration) of the larger Spatial Tracking
System, rearchitected to: build the map **live only** (no offline
compile stage -- the earlier two-round planning that proposed one was
explicitly rejected), generalize to **any indoor space** (no single-
room, single-floor, or Manhattan-orthogonality assumption), use a
**multi-process** live architecture (tracker/backend/dense as separate
OS processes; "no threads before P6" formally relaxed **only** for this
kind of process isolation), skip **AprilTags/fiducials** entirely, use
a **single approved tape-distance measurement** as the sole scale-
verification mechanism, and evaluate `nvblox` for the dense map under a
**strictly time-boxed, bounded** spike before considering Open3D CUDA.
Modules 3+4 (the STS repo) are unchanged and are the consumer this
work package's outputs (`walkable.json`'s schema, `calibration.json`'s
shape) are kept compatible with.

## 1. Honest status

**Everything in this document was built and, where the sandbox allows,
gate-tested in this development session.** This sandbox has no D435i,
no `pyrealsense2`, no GTSAM, and no GPU/`nvblox_torch` -- the same
constraints every `WP_*_Findings.md` in the base repo already lives
with. Where that matters, it's flagged explicitly below and in the
code's own docstrings, following this project's established discipline
of never silently presenting an unvalidated path as if it were proven.

**What IS validated in this sandbox**, via `tests/gates/test_g_live.py`
(**22/22 passing**) and the base repo's own `pyslam.selftest`
(**54/54 passing, unchanged** -- WP-LIVE's patches to existing files
introduced zero regressions):

- Every new pure-math module (plane fitting/classification/association,
  keyframe depth fusion, the generalized site frame, free-space-carved
  occupancy layers, the tape scale check, map-id hashing, the CPU
  voxel-hash dense fuser) via real oracle tests -- known-answer cases,
  not "it ran without crashing."
- The ICP fallback tracker, including a degeneracy detector, via a
  known-transform oracle **and** a corridor-degeneracy oracle.
- Local RGB-D relocalization (3D-3D RANSAC), including correct
  rejection of a geometrically-inconsistent candidate.
- Odometry's new PnP LM-refinement and prediction-gated matching,
  including a fail-open check (a deliberately bad prediction must not
  make tracking worse than not gating at all).
- **A full pipeline-level end-to-end check** that local RGB-D
  relocalization actually prevents a session break / replaces bridge
  links during a forced LOST -- and this check caught a real,
  previously-shipped bug (see 3.1 below) before it could reach
  hardware.
- **Real cross-process IPC**: shared-memory `FramePacket`/
  `PoseUpdatePacket` round-trips through an actual spawned
  `multiprocessing.Process`, not just in-process.
- **A full 3-process live-architecture run** (tracker, backend, dense,
  wired exactly as `run_live_map.py --mode live` would run them) over a
  real synthetic trajectory, producing real QA output and a real dense
  PLY export -- validated with `mp_start_method="fork"` because this
  specific sandbox container cannot rebuild POSIX semaphore state
  inside a `spawn`ed child (a container restriction, not a defect in
  this code -- see 4.1).
- **A full lockstep run** (tracker + the existing, unmodified
  `Pipeline` + dense, all in one process) producing a populated dense
  voxel map from a real trajectory.

**What is NOT validated here, and must be confirmed on the target
Orin/D435i before trusting it:**

- Everything touching `pyrealsense2` (the IMU-extrinsic fix in
  `realsense.py`, the capture-profile filter chain, `imu_capture_mode`
  interactions).
- The GTSAM-backed `Isam2Backend` (`pyslam/graph/backend_isam2.py`) --
  its logic reuses `backend_gtsam.py`'s already-oracle-tested
  conversion helpers, but its own incremental-update behavior (that
  repeated partial `update()` calls converge to the same answer a batch
  solve would) has not been run against real GTSAM anywhere in this
  session.
- `mp_start_method="spawn"` end-to-end (the documented production
  default) -- only `"fork"` was exercisable in this sandbox; see 4.1.
- Real timing/performance numbers anywhere. Several places are flagged
  below as NOT meeting the real-time frame budget yet (4.2), based on
  wall-clock measurements taken in this (slow, shared, non-Orin)
  sandbox -- treat them as "this needs profiling and likely
  optimization on real hardware," not as settled numbers.
- The `nvblox_eval.py` harness's own checks, beyond confirming they run
  and produce an honest "inconclusive, no GPU here" verdict rather than
  a fabricated one.

---

## 2. What changed / what's new, by file

### Patches to existing, already-gated files (all re-validated: 54/54)

| File | Change | Why |
|---|---|---|
| `pyslam/pipeline.py` | Local RGB-D relocalization tried before a LOST-recovery bridge; a real bug in the fall-through wiring found and fixed (3.1) | 4.1 (planning notes): local_reloc is the highest-priority online accuracy lever |
| `pyslam/frontend/odometry.py` | PnP LM refinement on the full RANSAC inlier set; optional prediction-gated matching (fails open) | 4.7: cheap, additive frontend accuracy/robustness |
| `pyslam/sensors/realsense.py` | IMU accel-to-colour extrinsic query fixed to use the callback-resolved profile, not the (accel-less, in callback mode) main pipeline profile | 3.5: this was silently falling back to IDENTITY on the project's own DEFAULT capture mode |
| `pyslam/graph/backend_native.py`, `pyslam/graph/backend_gtsam.py` | `local_reloc`-kind links get the same robust-kernel treatment as `loop`/`proximity` | Consistency; a single verified match still deserves robustification |
| `pyslam/core/config.py` | New `local_reloc_*` fields, opt-in, off by default | Same discipline every other opt-in in this file uses |

### New modules

```
pyslam/mapping/
  fused_depth.py     Level-1 per-keyframe depth fusion (weighted running mean, outlier rejection)
  planes.py           Generalized multi-plane RANSAC + classify (support/vertical/overhead) + associate
  site_frame.py        Site frame pinned to the LOCAL reference support plane (not global z=0)
  layers.py             Free-space-carved occupancy grid; exports STS's exact walkable.json schema
  dense_voxel.py         CPU voxel-hash point fuser (Level-2 fallback; DenseBackend Protocol for nvblox swap-in)
  scale_check.py          The approved single tape-distance scale verification
  lock.py                   Map bundle lock: map_id hashing, manifest, read-only
  qa_stream.py               Live mapping_status QA aggregation

pyslam/frontend/
  icp_fallback.py     Point-to-plane ICP tracking fallback, with a degeneracy detector

pyslam/loop/
  local_reloc.py      3D-3D RGB-D relocalization against recent working memory, for LOST recovery

pyslam/graph/
  backend_isam2.py    Incremental (iSAM2) pose-graph backend -- replaces "online + batch finalize()"

pyslam/live/           <- the multi-process live architecture
  capture_profile.py  One shared, hashable capture profile (mapping vs. runtime consistency)
  ipc.py                Shared-memory array marshalling + FramePacket/KeyframePointsPacket/PoseUpdatePacket
  tracker_stage.py       Odometry + ICP fallback + keyframe decision + fused depth + QA, factored for reuse
  backend_process.py      Reuses Pipeline UNCHANGED, fed via a QueueSensorSource (see 3.2 for the tradeoff)
  dense_process.py         Consumes tracker points + backend poses, integrates into the voxel fuser
  orchestrator.py            Spawns the 3 processes, wires IPC, clean shutdown
  lockstep.py                Single-process, deterministic path (selftest/replay/debugging)

pyslam/tools/
  nvblox_eval.py      Bounded (day-budgeted) nvblox_torch evaluation harness -- see 4.4

run_live_map.py        Main entrypoint: `--mode live` (multi-process) or `--mode lockstep`
scripts/tare_calibration_check.py   Pre-flight depth-scale check using the approved tape-distance mechanism
configs/capture_profile.mapping.json
tests/gates/test_g_live.py   22 real oracle/integration tests (run: python3 -m tests.gates.test_g_live)
```

---

## 3. Real bugs this work package's own gate discipline caught

Matching the base repo's own established culture (`WP_T_Findings.md`'s
mirrored-fixture bug, `WP_RELOC_Findings.md`'s Adjoint-direction
correction, etc.) -- every one of these was found by a test, not by
inspection, and every one is fixed and re-verified.

### 3.1 Stale bridge-link fall-through in `pipeline.py`

The pre-existing wiring for local_reloc-vs-bridge selection had the
bridge construction's shared `add_link`/`uf_union`/`record_adjacency`
tail **outside** the `if local_reloc succeeded / elif gyro / else
identity` branch entirely. A successful local_reloc added its own link
correctly, then unconditionally fell through and re-added whatever
`bridge_link` Python local a **prior loop iteration** had left bound --
silently duplicating stale links. Caught by
`check_local_reloc_prevents_session_break_end_to_end`, which asserts
link **counts**, not just presence (enabling local_reloc produced the
*same* bridge count as leaving it off, plus extra local_reloc links,
instead of trading one for the other). Fixed by moving the bridge
construction and its shared tail into an explicit `else:` block.

### 3.2 `[rho,phi]` Jacobian ordering bug in `icp_fallback.py`

The point-to-plane ICP Gauss-Newton step built its Jacobian in
`[phi,rho]` (rotation-first) column order while this codebase's frozen
`se3_exp`/`se3_log` convention is `[rho,phi]` (translation-first) --
the exact class of bug `pnp_info.py`'s own module docstring warns
about. It produced a plausible-looking pose with near-zero residual but
was structurally wrong. Caught by a known-transform oracle
(`check_icp_recovers_known_transform`), fixed, re-verified to
sub-millimetre/sub-thousandth-of-a-degree accuracy.

### 3.3 Sentinel-identity loss across a real process boundary

`backend_process.py`'s shutdown used a plain `object()` as an
end-of-stream sentinel, checked with `item is _SENTINEL`. This passed
every in-process manual test (no pickling occurs there) but silently
failed the instant it crossed a real `multiprocessing.Queue` --
pickling an `object()` produces a non-identical instance in the
receiving process, so the identity check never matched and the
sentinel was processed as if it were a real frame, crashing with
`AttributeError: 'object' object has no attribute 'rgb_handle'`.
Caught only by `check_orchestrator_full_three_process_pipeline_end_to_end`,
which is exactly why that test (spawning REAL OS processes, not mocks)
exists. Fixed by using `None` (a true CPython singleton, guaranteed to
survive pickling) as the sole sentinel.

### 3.4 Dense-process id-space mismatch

`dense_process.py` stores keyframe points keyed by the **tracker's**
`frame.frame_id`, while `backend_process.py` originally broadcast pose
updates keyed by the **backend's** own graph `node.id` -- a completely
independent counter (shared globally within one process across every
`extract_signature()` call, and in the live multi-process case, a
*separate* independent counter per process). The two numbering schemes
never coincided, so dense silently integrated nothing. Caught by
`check_lockstep_end_to_end_integrates_dense_map` (dense voxel count was
0 when it should have been thousands). Fixed by having the backend
translate `node.id -> frame.frame_id` via `PipelineResult.frame_records`
(which already records this correspondence) before broadcasting poses
to dense specifically; tracker's own pose feed stays `node.id`-keyed,
since tracker only needs a corrective nudge, not per-keyframe
correlation.

### 3.5 Missing final flush / shutdown race in `dense_process.py`

The dense process only exported its PLY on a periodic timer
(`export_interval_s`, default 5s); a session stopped shortly after
starting could exit before that timer ever fired once, producing no
output at all. Separately, the backend's own final pose broadcast
(sent *after* its Pipeline thread fully drains) could arrive after
dense's main loop had already exited on `stop_event`. Caught by
`check_orchestrator_full_three_process_pipeline_end_to_end` asserting a
PLY file actually exists after a short session. Fixed with a short
post-shutdown grace-drain window plus an unconditional final export.

### 3.6 `_run_pipeline`'s original `Pipeline.run()` call hid live state

`BackendProcessState` originally called `self.pipeline.run(source)`
directly, which only returns (and only then makes `PipelineResult`
visible to the caller) once the **entire** loop finishes -- fine for a
batch run, wrong for a live backend whose `poll_and_broadcast()` needs
to read the growing `PipelineResult.frame_records` **while** the loop
is still running (needed for 3.4's fix). Fixed by replicating `run()`'s
own short setup sequence directly and assigning `self._result` *before*
calling `_run_loop`, so the polling thread always sees the live object.
No change to `pipeline.py` itself was needed for this.

---

## 4. Known limitations, tuning items, and open work

### 4.1 `spawn` vs `fork` in this sandbox

`orchestrator.py` documents and defaults to `mp_start_method="spawn"`
for real deployment (fork silently duplicates device/GPU handles across
the fork boundary -- a well-known source of RealSense/CUDA crashes).
This specific development container cannot rebuild
`multiprocessing`'s POSIX semaphore state inside a spawned child
(`FileNotFoundError` from `SemLock._rebuild`) -- an environment
restriction, confirmed unrelated to this project's own logic by
successfully validating the identical 3-process wiring with `fork`
instead. **Confirm `spawn` works on the real Orin before trusting it
there**; if it turns out to have the same restriction (which would be
surprising), `--mp-start-method fork` is available as a fallback,
accepting fork's device-handle-duplication risk.

### 4.2 Real-time budget not yet met by the ICP fallback

`icp_fallback.py`'s correspondence search is a coarse voxel-bucket
lookup in pure Python, not a real k-d tree. At the point counts needed
for a well-conditioned scene it currently costs on the order of a
second or more per call **in this (slow, shared) sandbox** -- nowhere
near the tracker's frame-rate budget. `tracker_stage.py` already
mitigates this by using a coarse stride (16px) and a hard point cap
(1500) when building the query cloud for this specific call site, which
is a real, working mitigation, but a proper fix (vectorized
correspondence search, or a real k-d tree) is needed before this path
can be trusted at 30Hz on the Orin. **Profile on real hardware before
relying on this in a live run**; until then, `TrackerStage`'s ICP
fallback is best understood as "correctness-validated, performance-TBD."

### 4.3 Backend recomputes its own odometry (accepted tradeoff)

`backend_process.py` and `lockstep.py` both reuse the existing,
unmodified `Pipeline` class as the backend's brain, fed via IPC/direct
frame replay instead of a camera. This means `Pipeline`'s own
`VisualOdometry` runs a **second, independent** pass over every frame,
duplicating the tracker's frontend cost. This was a deliberate scope
decision (see `backend_process.py`'s own module docstring): it reuses
100% of the already-validated, already-gated memory/retrieval/verify/
local_reloc/proximity/gravity-prior/graph logic with zero risk of a new
correctness bug, instead of a hasty mid-project refactor of
`pipeline.py`'s internals to accept externally-decided keyframes from
the tracker. **The natural WP-LIVE follow-up** (not done here, tracked
as open) is giving `Pipeline` an injectable "keyframe decision already
made upstream" mode so the backend can skip its own frontend pass
entirely -- this is where the real performance win from the
tracker/backend split is still waiting to be realized.

### 4.4 nvblox evaluation is genuinely incomplete

`pyslam/tools/nvblox_eval.py` is the bounded harness, not the
evaluation itself -- it has never run against real `nvblox_torch` or a
GPU. Every check in it returned "skipped, not importable here." **Run
it on the target Orin Nano** (`python3 -m pyslam.tools.nvblox_eval
--day-budget 7`, re-running across up to 7 calendar days as the budget
allows) to get a real go/no-go. Until then, `dense_process.py` uses
`VoxelHashFuser` (the CPU fallback), which works but is explicitly
**not** equivalent to a real TSDF/marching-cubes mesh -- see
`dense_voxel.py`'s own docstring for exactly what it doesn't do
(no signed-distance field, no watertight surface, no free-space
integration for surface refinement).

### 4.5 Live-mode map locking is incomplete

`run_live_map.py --mode lockstep`'s `_lock_and_write` is fully wired
(writes the dense PLY, computes `map_id`, writes a manifest). `--mode
live`'s equivalent is **not** -- the multi-process architecture
deliberately keeps the backend's live `Pipeline`/graph object
out-of-process, so there's no simple way to serialize a full manifest
(map_id, per-plane-landmark summary, trajectory) back across the
process boundary at shutdown today. `run_live_map.py` writes a
`session_summary.json` flagging this explicitly rather than silently
producing an incomplete-but-unlabeled bundle. **The natural fix**: have
the backend process periodically write its own trajectory/graph
snapshot to disk (reusing the base repo's existing
`pyslam/tools/trajectory_export.py`, unchanged) rather than trying to
serialize the whole `Pipeline` object back to the orchestrator.

### 4.6 Plane landmarks are built but not yet wired into the live loop

`pyslam/mapping/planes.py` (fit/classify/associate) and
`pyslam/mapping/site_frame.py` (site frame from a reference support
plane) are both built and oracle-tested standalone, but **nothing in
`tracker_stage.py`/`backend_process.py`/`lockstep.py` calls them yet**.
Today's site frame construction in `_lock_and_write` passes
`reference_plane=None` and says so honestly in the manifest. Wiring
plane extraction into the per-keyframe backend loop (extract from each
keyframe's fused-depth points, associate against existing landmarks,
feed the nearest support plane to `site_frame.py`) is the natural next
WP-LIVE step -- the pieces exist and are tested; the integration does
not yet.

### 4.7 Dense submap granularity is one keyframe = one submap

`dense_process.py` integrates each keyframe's points as its own
"submap" (simplest possible mapping). The planning notes' Level-2
design describes grouping several consecutive keyframes into
metre-scale submaps for voxel-hash memory efficiency at facility scale
-- not required for correctness, and not done here. A large, long
capture would benefit from this before relying on `VoxelHashFuser`'s
memory footprint staying bounded.

### 4.8 No paging for facility-scale captures

`DenseProcessState._max_stored` (unbounded by default) and the base
repo's own WM/LTM memory management exist for exactly this class of
problem but aren't connected for the Level-1 point-cloud store here.
Flagged, not built -- a room-to-building-scale capture is fine; a
genuinely large, multi-session facility capture would need this.

---

## 5. Run it yourself, right now

```bash
cd pyslam_live   # wherever you extracted the zip
python3 -m pyslam.selftest              # 54/54, base repo, unchanged -- run this first always
python3 -m tests.gates.test_g_live      # 22/22, WP-LIVE's own gate suite, hardware-free
python3 -m pyslam.tools.nvblox_eval --day-budget 7   # honest "no GPU here" on a dev machine
```

`git log`-equivalent for this session doesn't exist (this was built in
a single session against a zip, not an incrementally-committed repo) --
this document, the inline module docstrings, and `test_g_live.py`'s own
test names are the record of what was built, why, and what it proves.

See `RUNBOOK_LIVE.md` for the hardware bring-up sequence (extends
`README_ORIN.md`, doesn't replace it) and `requirements_live.txt` for
what's new on top of the base repo's dependencies.
