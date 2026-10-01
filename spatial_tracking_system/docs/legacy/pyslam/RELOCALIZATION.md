# Relocalization

Single-snapshot 6DOF global relocalization: point the D435i at
something, get back where the camera is inside a map an earlier
`run_slam.py` run built. Monocular at query time -- the map's own
depth (recorded during mapping) is what makes the reported pose
metric, not the query's.

Read `Runbook.md` first if you haven't; this assumes a working
`run_slam.py` setup. See `WP_RELOC_Findings.md` for what's validated
and what isn't, mirroring `WP_J_Findings.md`'s role for the Orin port.

---

## What changed, in one paragraph

A finished mapping run stores poses (`keyframes.json`, `graph.g2o`,
`map.ply`) but never stored appearance -- the ORB descriptors and 3D
points a relocalizer needs lived only in RAM and in an LTM SQLite file
`run_slam.py` never even named, so it landed in `/tmp` under a random
name. Every mapping run now also writes `<run_dir>/reloc_map/`, a
small, self-contained, read-only pack holding exactly what
`relocalize.py` needs. Four existing files gained small, additive
changes (`run_slam.py`, `pyslam/tools/run_outputs.py`,
`pyslam/loop/verify.py`, `pyslam/core/config.py`); nothing in
odometry, memory, the graph, or the Bayes filter changed.

## Quick start

```
# one command covers everything: mapping run now writes reloc_map/ too
python3 run_slam.py --realsense --imu

# plug the camera back in, get a pose
python3 relocalize.py --map runs/run_1758.../reloc_map
```

Exit code is `0` on `LOCALIZED`, `2` otherwise -- scriptable.

### Everything else

```
# replay a saved snapshot against the same map (no camera needed)
python3 relocalize.py --map runs/run_.../reloc_map --image query.png

# diagnosis and tuning
python3 relocalize.py --map runs/run_.../reloc_map \
    --burst 7 --topk 20 --save-query --overlay --depth-crosscheck \
    --out reloc_out/ --config-override reloc_min_inliers=30

# best-effort retrofit of a run from BEFORE this feature existed
python3 -m pyslam.tools.build_reloc_map runs/run_1758...
```

`--map` is the only required argument.

## Gate suite

```
python3 -m tests.gates.test_greloc
```

Six checks: a transform-chain oracle (recovers a known query pose to
sub-centimetre accuracy), false-positive rejection against an
unrelated scene, a map-pack round-trip, a unit check of the
spatial-coverage gate, a Monte Carlo covariance/NEES oracle specific
to this feature's own pose composition (not a re-test of
`pnp_info_matrix` itself -- `test_g0.py` already covers that function;
this checks the *composition* relocalize.py builds on top of it), and
a sparse-map (single-node) localization check. All six passed during
development; see `WP_RELOC_Findings.md` for the actual numbers,
including one real bug this oracle caught before it shipped (below).

## How it works

1. **Load the pack.** mmap the descriptor/global-descriptor arrays,
   open the node SQLite table `mode=ro` (a real read-only file
   descriptor, not a promise the code keeps on its own honour), read
   `meta.json`. Warn loudly on an intrinsics or config-hash mismatch.
2. **Capture.** `SnapshotSource` owns the D435i for exactly as long as
   it takes: discard warm-up frames while auto-exposure settles, burst
   a few frames, keep the sharpest by variance-of-Laplacian. Reject
   outright if the whole burst is below the blur floor -- motion blur
   is the most common cause of a spurious not-found on live hardware.
3. **Extract.** The same `make_orb` + `grid_bucket` stage a mapping
   run uses, minus depth association -- the query Signature carries
   all-NaN `kp3d` and an all-False `valid` mask by construction.
4. **Shortlist.** One cosine matvec of the query's global descriptor
   against every packed node (`reloc_shortlist_backend=global_desc`,
   the default). If nothing clears the "new place" baseline
   (`vpr/likelihood.normalize_likelihood`), stop here -- NOT_FOUND
   without ever touching PnP.
5. **Verify.** Best-first through the shortlist: match descriptors,
   solve PnP+RANSAC, apply every gate below.
6. **Decide.** Two independently-verified candidates must agree
   (cross-candidate consensus) or, on a sparse map where only one
   verifies, it must clear a stricter single-candidate bar. See
   "Rejection gates" below.
7. **Report.** `reloc_result.json`, an append to `reloc_log.jsonl`,
   optionally an overlay PNG. Never a bare pose.

### The transform chain (read this before changing anything here)

A map node `n` stores `kp3d` in **its own camera frame** and
`pose_map[n]`: world←n. `solve_pnp_ransac(obj_pts=node's kp3d,
img_pts=query's kp, K=query's K)` returns the raw OpenCV convention
`T_cam_obj`, which here is directly `T_query_node` (query-camera←node,
no inversion needed -- the same "direction 2" shape
`loop/verify.py`'s own bidirectional check uses).

```
T_world_query = pose_map[n] @ inverse(T_query_node)
```

For the covariance: `pose_map[n]` is a fixed, already-known **left**
constant in that composition, so a right-tangent perturbation of
`inverse(T_query_node)` induces the *identical* right-tangent
perturbation of `T_world_query`, with no further Adjoint needed (the
same argument `pnp_info_matrix`'s own docstring makes for
`odometry_f2m.py`'s case). Since `T_world_query`'s uncertainty is
therefore `inverse(raw PnP output)`'s uncertainty -- the "direction 1"
shape, not "direction 2" -- the correct function is `pnp_info_matrix`
(with its Adjoint step), **not** `pnp_info_matrix_direct_frame`.

**This is a correction, recorded honestly:** the original architecture
plan for this feature said the opposite -- `pnp_info_matrix_direct_frame`,
no Adjoint. Working through the actual composition while implementing
it (not while planning it) showed that was backwards. Getting it wrong
would have produced a covariance off by an Adjoint transform: plausible
numbers, wrong meaning, and nothing about the gate that uses them would
have looked broken from the outside. `check_covariance_oracle_reloc_frame`
in the gate suite exists specifically to catch this class of mistake by
Monte Carlo rather than by re-reading the algebra a third time -- same
discipline `pnp_info.py`'s own module docstring describes for exactly
this kind of bug, and the same discipline that caught it here (mean
NEES landed at 6.29 against an expected ~6.0; an Adjoint-direction bug
of this shape typically shows up as ~24, not a near miss).

### Metric scale without query depth

The map's 3D points are metric because they came from the D435i's own
depth at *mapping* time. A depthless query's 2D-3D PnP against them is
therefore fully constrained in all six degrees of freedom, in real
units, with no scale ambiguity. This is the one fact the whole design
depends on -- the query intentionally never touches its own depth
stream for the pose solve. (`--depth-crosscheck` reads live depth
anyway, purely as an independent diagnostic; see below.)

## Rejection gates

Robust means: would rather say nothing than something wrong. Every
gate is independent; all must pass.

| Gate | Default | What it actually catches |
|---|---|---|
| Retrieval standout | likelihood > baseline | Featureless/aliased scenes where every candidate scores alike |
| Inlier count | `reloc_min_inliers=25` | Above `verify_min_inliers=20` (the loop-closure value) -- a monocular query is a weaker constraint than a verified two-way loop |
| Inlier ratio | `reloc_min_inlier_ratio=0.35` | Diffuse mismatches with enough raw inliers to look plausible |
| Reprojection RMS | `reloc_reproj_px=3.0` | A converged-but-strained solution |
| Grid-cell spread | `reloc_min_grid_cells=6` of 48 | A spatial **coverage floor** -- inliers clustered on one small image region (a poster, an object). **Not** a 3D-planarity test: a fronto-parallel wall filling the whole frame clears this easily, confirmed empirically (see below) |
| Covariance | `reloc_max_sigma_trans_m=0.5` | The actual ill-conditioning gate -- reads `pnp_info_matrix`'s real Hessian, not a heuristic. This is what would catch a genuine bas-relief-style degeneracy, not the grid-cell gate |
| Cross-candidate consensus | `reloc_consensus_trans_m=0.25` / `reloc_consensus_rot_deg=5.0` | The monocular stand-in for `verify.py`'s bidirectional PnP check, which a depthless query cannot run |
| Retrieval margin | `reloc_retrieval_margin=1.3×`, waived for graph neighbours | Perceptual aliasing between genuinely different places. Only hard-blocks the riskiest path: a single verified candidate under an ambiguous retrieval margin is downgraded from LOCALIZED to DEGENERATE, rather than blocking every localization outright |

**On the grid-cell gate specifically:** the architecture plan for this
feature originally described it as catching "the classic single-planar-
patch case." Building the gate suite's degenerate-case test found this
overclaimed what a 2D pixel-space spread check can actually detect --
perspective gives real depth variation across a fronto-parallel wall
filling the frame, so that case passes the grid-cell gate easily (and
correctly: PnP genuinely isn't degenerate there). The comment in
`pyslam/loop/relocalize.py` and the config docstring were corrected to
describe this gate honestly, as a coverage floor against small-region
clustering. The covariance gate is what actually catches ill-
conditioned geometry.

**Terminal states**, never a bare pose:

- `LOCALIZED` -- pose, covariance, `consensus: "dual"` or `"single"`
- `AMBIGUOUS` -- two candidates verified but disagreed
- `DEGENERATE` -- geometry passed base gates but conditioning (or the
  single-candidate bar, or an ambiguous retrieval margin on a
  single-candidate accept) failed
- `NOT_FOUND` -- with the stage it stopped at

Every failure is diagnosable from `reloc_result.json` alone.

## Sparse-map path (80-keyframe maps)

At the floor this feature targets, a query viewpoint may genuinely
overlap only one keyframe -- requiring two independent verifications
would then reject correct localizations for a reason unrelated to
correctness. A single verified candidate is still accepted, but only
under a stricter bar: `reloc_single_candidate_min_inliers=40` (vs. 25),
`reloc_single_candidate_min_grid_cells=10` (vs. 6), and
`reloc_single_candidate_max_sigma_trans_m=0.25` (vs. 0.5). The result
carries `consensus: "single"` so a downstream consumer always knows
which evidence path produced the pose. Validated in the gate suite
(`check_sparse_map_single_candidate`) against a genuine one-node map.

## Retrofitting an existing run

`pyslam/tools/build_reloc_map.py` is **salvage, not a guaranteed
reconstruction** -- said plainly because both sources it can pull from
are incomplete by construction:

1. **The imagery side-cache**, if the run used `--keep-imagery-cache`.
   Nodes found there get features re-extracted from the cached RGB+
   depth PNGs using the run's own `config.json` -- byte-identical to
   what a live run would have packed.
2. **Orphaned LTM SQLite files** in the system temp directory (every
   run before this feature's `run_slam.py` change dumped its LTM to a
   randomly-named `tempfile.mkstemp()` path nothing pointed back to).
   Matched to this run by node id **and** timestamp, so a coincidental
   id collision between two unrelated runs is implausible in practice.

Coverage is reported honestly in `meta.json`'s `retrofit_coverage`
block -- a keyframe recovered by neither route is simply absent from
the pack, and `relocalize.py` will never propose it as a candidate.
That is the safe failure mode: a smaller map that still localizes
correctly against what it does have, not a map that pretends to be
complete. Check before relying on a retrofit:

```
ls runs/*/imagery_cache 2>/dev/null
python3 -c "import json; print(json.load(open('runs/RUN/map_stats.json'))['n_nodes_from_imagery_cache'])"
```

## Read-only by construction, not just by convention

`RelocMap` opens its node table with `sqlite3.connect(f"file:{path}?
mode=ro", uri=True)` -- a real file-descriptor-level restriction; any
write attempt raises. `relocalize.py` never calls `pipeline.graph`,
never inserts a node, never touches `--map` at all. No merging, no
optimisation, strictly one query pose against a frozen map.

## Performance (Jetson Orin Nano 8GB, `nvpmodel -m 0` + `jetson_clocks`)

Estimated, not yet measured on your specific board -- `relocalize.py
--bench` prints real per-stage timing on every query; replace the
numbers below with yours after the first run.

| Stage | 80 nodes | 300 nodes |
|---|---|---|
| Pack load (once per process) | 40-70 ms | 80-150 ms |
| Warm-up + burst capture (once) | 0.5-1.0 s | 0.5-1.0 s |
| ORB + grid bucket | 18-28 ms | 18-28 ms |
| Global-descriptor shortlist | 2-3 ms | 2-4 ms |
| Match + PnP per candidate | 20-35 ms | 20-35 ms |
| **Per-query total** | **~150-250 ms** | **~150-250 ms** (2-3 candidates typically reach geometry) |

Peak resident memory: roughly 300 MB Python/NumPy/OpenCV/pyrealsense2
baseline, plus 4-16 MB of map pack. All CPU; the GPU is never touched.

Exact brute-force matching (`reloc_shortlist_backend=brute`) scales
linearly with total descriptors in the map and is kept only as a
ground-truth comparison for validating the default shortlist's recall
-- expect roughly 0.3-0.8s at 80 nodes, several seconds at 300+.
`hnswlib` is deliberately not used anywhere in this feature: at the
node counts this targets, an exact cosine matvec is already
microseconds, so approximate search would be dependency risk for zero
measurable gain.

## Capturing a map that relocalizes well

At 80 keyframes, viewpoint coverage is the limiting factor, not
compute -- a query only succeeds if it falls inside some keyframe's
ORB overlap cone (roughly a metre of translation, thirty degrees of
rotation, less on low-texture surfaces).

- Sweep the camera while mapping rather than pointing it straight
  ahead the whole way; a corridor walked facing forward only
  relocalizes from facing forward.
- Revisit places. Loop closures densify coverage where they fire and
  improve the poses the pack inherits.
- Start with the runbook's two-second still hold. Without it, gravity
  alignment fails and `W_grav` output degrades to the first camera's
  optical axis rather than true vertical -- reported honestly in
  `meta.json`, but still not what you want.
- **A map's own pose-graph accuracy is a ceiling on relocalization
  accuracy, not a floor `relocalize.py` can rescue.** Relocalization
  reports a pose consistent with the map's *own* internal frame,
  wherever that frame actually is -- if a run had no loop closures to
  correct accumulated odometry drift, the map's own coordinates can be
  meaningfully offset from true world coordinates even though every
  individual query localizes correctly and consistently *within* that
  map. (Observed directly during validation: an un-loop-closed
  synthetic run's `pose_map` for one keyframe had drifted 2.35 m / 142°
  from its true pose -- and `relocalize.py` still recovered a query
  pose within 1.9 mm of what the map's own frame said was correct.
  That's the relocalizer doing its job right; it's the map's own
  accuracy that needed the loop closures it didn't get.)

## Changes to existing files

| File | Change |
|---|---|
| `run_slam.py` | `Pipeline(..., ltm_path=<run_dir>/ltm.sqlite3)` instead of an unnamed `/tmp` file |
| `pyslam/tools/run_outputs.py` | New step 3.5: writes the map pack after `finalize()`, same try/except-per-step discipline as every other step |
| `pyslam/loop/verify.py` | Public aliases `match_descriptors`/`solve_pnp_ransac` for the existing `_match`/`_solve_pnp` -- no behaviour change |
| `pyslam/core/config.py` | New `reloc_*` block plus a `CONFIG_CHOICES` entry for `reloc_shortlist_backend` |

Nothing in `odometry.py`, `odometry_f2m.py`, `memory.py`, `posegraph.py`,
or `bayes.py` changed. The existing gate suite's meaning is unaffected.
