# WP-RELOC Findings

What's validated, what isn't, and what changed between planning this
feature and building it. Read alongside `RELOCALIZATION.md` (the "how");
this is the "what to trust," same split `WP_J_Findings.md` uses for the
Orin port.

## Validated

All six checks in `tests/gates/test_greloc.py` pass (`python -m
tests.gates.test_greloc`):

```
transform-chain oracle: 3.65mm / 0.160deg (consensus=dual)
false-positive rejection: 0/5 false accepts against an unrelated room
pack round-trip: 5 nodes, descriptors/poses/adjacency bit-identical
grid-cell coverage gate: clustered=1 cells, spread=38 cells, threshold=6
covariance oracle: mean NEES=6.29 over 500 trials (want ~6.0)
sparse-map (1-node) single-candidate path: LOCALIZED, error=2.67mm
```

Also validated by hand, outside the gate suite, against a real
`Pipeline` run (not a hand-built fixture): `run_slam.py`'s own
`finalize_and_export` → `write_map_pack` → `RelocMap` →
`relocalize()` path, end to end, on a 24-keyframe synthetic
`square6dof` run. See "The odometry-drift finding" below for what that
run surfaced.

## A correction, made while implementing rather than while planning

The architecture plan for this feature (written before any code)
specified `pnp_info_matrix_direct_frame` for the reported pose's
covariance -- no Adjoint step. Deriving the actual composition while
writing `pyslam/loop/relocalize.py` showed this was backwards.

The reported pose is `T_world_query = pose_map[n] @
inverse(T_query_node)`, where `T_query_node` is the raw PnP output.
`pose_map[n]` is a fixed, already-known **left** constant in that
composition, so (the same argument `pnp_info.py`'s own docstring makes
for `odometry_f2m.py`'s case) a right-tangent perturbation of
`inverse(T_query_node)` induces the identical right-tangent
perturbation of `T_world_query`. Since `T_world_query`'s uncertainty is
therefore the uncertainty of an *inverted* raw PnP solve -- the
"direction 1" shape in `loop/verify.py`'s own vocabulary, not
"direction 2" -- the correct function is `pnp_info_matrix` (with its
Adjoint step), not `pnp_info_matrix_direct_frame`.

Getting this backwards would not have crashed anything or looked wrong
from the outside: it produces a covariance that is off by an Adjoint
transform, i.e. plausible numbers with the wrong meaning, silently
feeding the `reloc_max_sigma_trans_m` gate a value that doesn't
describe the actual uncertainty. This is exactly the class of bug
`pnp_info.py`'s own module docstring warns about ("an earlier version
of this derivation got the adjoint's placement wrong... caught by the
Monte Carlo oracle, not by inspection") -- and it's why
`check_covariance_oracle_reloc_frame` exists as a dedicated Monte Carlo
check of *this feature's own composition*, not a re-test of
`pnp_info_matrix` in isolation (`test_g0.py`'s
`test_pnp_info_matrix_oracle` already covers that function). Mean NEES
landed at 6.29 against an expected ~6.0; the specific bug this would
have caught (wrong Adjoint direction) produces NEES around 24 in
`test_g0`'s own precedent for the same mistake, not a near miss -- so a
6.29 is a real pass, not a coincidence in the noise.

## A second correction: what the grid-cell gate actually catches

The architecture plan described `reloc_min_grid_cells` as catching
"the classic single-planar-patch case... depth axis probably
underconstrained." Building the gate suite's degenerate-case test
(an attempt to construct a near-planar, fronto-parallel single-wall
view and confirm it gets rejected) found this overclaimed what a 2D
pixel-space spread check can detect.

Measured directly: a synthetic camera positioned 3.5m from a flat wall,
facing it fronto-parallel, produced depth values spanning only ~4cm
across a ~1.49m mean distance (std/mean ≈ 2.8%) -- about as close to
"planar" as a real scene gets -- and still localized cleanly, with
`sigma_trans=3.5mm` and 413/454 inliers spread across 45 of 48 grid
cells. Perspective alone gives real depth variation across a
frame-filling flat surface; a 2D coverage check has no way to
distinguish that from genuinely well-distributed 3D geometry.

The gate is still useful -- it catches inliers clustered on one small
image region (a poster, a single object), confirmed directly in
`check_grid_cell_gate` (a 40×40px corner cluster scores 1 cell against
a threshold of 6; a full-frame spread scores 38) -- but it is not a
3D-planarity or bas-relief-ambiguity test. That job belongs to the
covariance gate, which reads the actual PnP Hessian rather than pixel
positions. Both the code comment in `relocalize.py` and the
`Config.reloc_min_grid_cells` docstring were corrected to describe this
honestly rather than leave the overclaim in place. No attempt was made
to construct a synthetic scene that actually defeats the covariance
gate (a genuine bas-relief case needs a narrow field of view and a
long viewing distance relative to depth variation, neither of which
this project's synthetic room fixtures or the D435i's ~54° horizontal
FOV make easy to hit) -- flagged here as not attempted, not silently
assumed safe.

## The odometry-drift finding (not a relocalization bug)

While validating against a real `Pipeline` run rather than a hand-built
fixture, an early test appeared to fail: querying near a mapped node's
`pose_map` (offset by a few centimetres) returned NOT_FOUND, and even
direct verification against the correct node produced almost no
inliers.

Diagnosis: that specific synthetic run (`square6dof`, default `Config`,
120 frames, **zero loop closures**) had accumulated real odometry drift
-- the queried node's `pose_map` differed from its true ground-truth
camera pose by **2.35m translation and 142° rotation**. The test's
query was built by offsetting `pose_map[n]` by a few centimetres and
then rendering that pose against the *true* synthetic world geometry --
so the "nearby" query was actually nowhere near the true camera
position that node's own stored 3D points came from. Re-run using the
node's true ground-truth pose (`result.node_gt[n] @ T_BODY_CAM`)
instead: `relocalize()` recovered a pose within **1.9mm** of what the
map's own (drifted) internal frame said was correct, with three
independent candidates (not just two) verifying and agreeing.

This is the relocalizer working correctly, not a bug: its job is to
report a pose consistent with the map's own coordinate frame, whatever
that frame's own accuracy is. It surfaces a real, separate fact worth
knowing when planning a mapping run: **a map with no loop closures can
have its own internal frame meaningfully offset from true world
coordinates**, even though every individual relocalization query
against it will still be internally consistent and accurate. This is
recorded in `RELOCALIZATION.md`'s "capturing a map that relocalizes
well" section as operational guidance (revisit places; loop closures
are what keep the map's own frame close to the true one), not treated
as something for `relocalize.py` itself to fix -- it has no way to know
the map's own frame is off, and shouldn't invent one.

## Not attempted

- A genuine bas-relief-ambiguity synthetic fixture that defeats the
  covariance gate specifically (see above).
- Hardware validation on the actual Jetson Orin Nano / D435i rig --
  everything above is synthetic-fixture and real-Pipeline-but-synthetic-
  source validation. `RELOCALIZATION.md`'s performance table is
  estimated from the mapping pipeline's own measured Orin timings
  (`README_ORIN.md`, `WP_J_Findings.md`), not measured for this
  feature directly. `relocalize.py --bench` exists specifically so the
  first real run replaces those estimates with real numbers.
- `reloc_shortlist_backend="brute"` was exercised only for a handful of
  nodes in testing, not at the 80-300 node range its own sizing note in
  `RELOCALIZATION.md` describes; the linear-scaling estimate there is
  arithmetic from the shortlist's per-candidate cost, not a measurement.
