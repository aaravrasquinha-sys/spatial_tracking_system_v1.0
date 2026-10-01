# pyslam — M1 (master map) + M2 (static-camera anchor)

**Never imports** `poi_*` or `sts`. Imports only itself.

| | M1 live mapper | M2 anchor |
|---|---|---|
| Consumes | RealSense D435i (live or `.bag`), capture profile | map bundle + a static capture of the camera seated in its mount |
| Produces | `data/maps/<site>/dense/points.ply`, `capture_profile.json`, `session_summary.json` | `data/anchors/<site>/<cam>/…` ([contract](../contracts/anchor_bundle.md)) |
| Entry points | `run_live_map.py` (live multi-process), `run_bag.py`, `run_slam.py`, `run_synth.py`, `relocalize.py` | `run_anchor.py prepare · capture · solve · calibrate · markers · check` |
| Through `sts` | `python -m sts map`, `map-lock` | `python -m sts anchor <verb>`, `accept`, `check` |
| Config | `pyslam.core.config.Config` (`--config-override k=v`), `CaptureProfile` | `pyslam.anchor.config.AnchorConfig` (`--set k=v`; `site.json -> anchor.set`) |
| Tests | `python -m pyslam.selftest` (54), `python -m tests.gates.test_g_live` (22), `test_g0/2/3/4/5/a/l/m/traj/reloc` | `python -m tests.gates.test_g_anchor` (20) |
| Docs | `docs/legacy/pyslam/` (`SYSTEM_SUMMARY_LIVE.md`, `RUNBOOK_LIVE.md`, `README_ORIN.md`, `RELOCALIZATION.md`, `WP_*_Findings.md`) | `docs/legacy/pyslam/SYSTEM_SUMMARY_ANCHOR.md`, `RUNBOOK_ANCHOR.md` |

## M2 in one paragraph
3D-to-3D registration, not monocular relocalisation. The camera's accelerometer gives roll/pitch and its own floor RANSAC gives
height (no map needed), leaving x, y, yaw, which an FFT correlative search finds exhaustively (so **ambiguity is measurable**:
symmetric rooms tie). Robust multi-scale point-to-plane ICP refines from every distinct peak. Independent gates (ICP RMSE and
fitness, IMU vs map roll/pitch, own-floor vs map height/tilt, uniqueness, observability, sigma, repeatability, depth residual)
must all pass; otherwise the result is written as `.REJECTED.json` and M4 cannot load it.

## Limits (docs/OPEN_ITEMS.md)
H1 not validated on hardware · H4 capture profile declared not enforced · H5 room must be empty · H6 no second global method ·
K1 lockstep locking undefined · K2 hardcoded fork-mode intrinsics · K3 fork is the script default · paused: gyro bridge,
f2m parity, corridor loops, IMU fusion.

## Editing it
Keep the map bundle and anchor bundle contracts. Inside the package you are free: the gate tests are oracle tests with known
answers, plus mutation checks (`pyslam/tools/mutation_check.py`). Run `sts test selftest g_live g_anchor` before and after.
