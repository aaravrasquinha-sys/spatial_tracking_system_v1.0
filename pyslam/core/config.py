"""
Single frozen config object. Hashed into every run directory so every
output is traceable to the exact settings that produced it.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
import hashlib
import json


@dataclass(frozen=True)
class Config:
    # --- features ---
    n_features: int = 1000
    grid_cols: int = 8
    grid_rows: int = 6
    orb_scale_factor: float = 1.2
    orb_n_levels: int = 8

    # --- depth association ---
    depth_min_m: float = 0.3
    depth_max_m: float = 4.0
    depth_patch: int = 3          # (2k+1)x(2k+1) median patch
    depth_grad_max_m: float = 0.15  # reject if local depth gradient exceeds this

    # --- odometry ---
    keyframe_trans_m: float = 0.15
    keyframe_rot_deg: float = 15.0
    keyframe_min_inliers: int = 60
    odom_reproj_px: float = 3.0
    odom_min_inliers: int = 15

    # WP-B1: frame-to-local-map odometry (odometry_f2m.py)
    odometry_backend: str = "f2f"  # "f2f" (Phase 0) or "f2m" (WP-B1)
    f2m_max_landmarks: int = 2000       # RTAB-Map's own F2M default
    f2m_guided_window_px: float = 25.0  # projection-guided match search radius
    f2m_min_inlier_ratio: float = 0.5  # plausibility gate (Phase 1 plan F3).
        # WP-B1 continued: raised from the original 0.35 after a
        # per-frame diagnostic trace found seed-2's square6dof run
        # accumulating map_ATE in discrete JUMPS that each correlated
        # directly with a keyframe whose inlier_ratio sat in the
        # 0.35-0.5 band (e.g. inlier_ratio~0.4 at frame 45 -> error
        # jumped from ~1cm to ~8cm immediately after; a similar event
        # near frame 85 -> jump to ~17cm) -- RANSAC still accepted these
        # as its best solution, and the plausibility gate at 0.35 let
        # them through as if they were trustworthy, after which
        # `_add_landmarks` permanently baked the resulting slightly-off
        # pose into new map landmarks (this tracker's own map has no
        # mechanism to later un-bake that -- see local_ba.py's
        # deliberately-conservative-not-aggressive design). A 3-seed,
        # 3-value sweep (0.35/0.5/0.6) confirmed 0.5 as the right
        # tightening: seed=2's map_ATE nearly halved (17.66cm ->
        # 9.86cm) with ZERO change to seeds 1/3 (both hit the exact
        # same map_ATE bit-for-bit at 0.35 and 0.5, confirming this
        # only screens out genuinely marginal keyframes, not ordinary
        # ones); 0.6 was tested and REJECTED -- it starts rejecting
        # keyframes RANSAC and the OTHER plausibility checks would
        # otherwise have accepted, causing real LOST events (0/0/0 ->
        # 1/1/3 across the three seeds) and a large ATE regression even
        # on seeds 1/3 that 0.5 left untouched. See WP_B1_Findings.md
        # for the full sweep and the per-frame trace that motivated it.
    f2m_min_spatial_spread: float = 0.15  # min fraction of image width/height inliers must span
    f2m_motion_gate_factor: float = 8.0   # Mahalanobis-style bound vs constant-velocity prediction
    f2m_keyframe_coverage_floor: float = 0.25  # force a new keyframe if tracked/matched ratio drops this low
    f2m_ba_window: int = 5  # WP-B1 finish: sliding-window local bundle
        # adjustment window size in keyframes (local_ba.py). Matches the
        # Phase 1 plan's own "~5 keyframes" sizing for this item.
    f2m_fusion_radius_m: float = 0.05  # WP-B1 continued: landmark
        # data-association radius (odometry_f2m.py::_add_landmarks). A
        # candidate keypoint that failed to MATCH an existing landmark
        # this frame (guided/unguided both use a 0.8 Lowe-ratio test,
        # which starves under near-duplicate descriptors -- see
        # WP_B1_Findings.md's duplicate census: 100% of a 300-landmark
        # sample had >=1 near-duplicate before this fix) is still
        # checked against the spatial neighbourhood of the existing map
        # before being inserted as brand-new. 5cm chosen as tight enough
        # not to merge genuinely distinct nearby points (ORB features on
        # this project's synthetic wall texture are typically >10cm
        # apart at the depth ranges this fixture set uses) while wide
        # enough to catch the same physical point re-observed with a
        # few cm of pose-estimate noise between keyframes.
    f2m_fusion_hamming_max: int = 50  # WP-B1 continued: descriptor
        # agreement required (out of 256 bits) for a spatially-close
        # candidate to be treated as the SAME landmark rather than a
        # coincidentally-nearby different one. Deliberately tighter than
        # P3's cross-frame incremental-vocabulary threshold (70, see
        # p3_new_word_hamming_threshold below) -- landmark fusion is a
        # stronger claim ("this IS that point") than "these two
        # descriptors should share a vocabulary word", so it should be
        # harder to satisfy. Not independently swept against
        # aliasing_rooms's false-fusion risk yet; see WP_B1_Findings.md
        # section on landmark fusion for what a proper calibration sweep
        # (mirroring P3's own Hamming-threshold sweep) would look like.
    lost_consecutive_frames: int = 5

    # --- memory ---
    stm_size: int = 10

    # --- Phase 2 (WP-P2): LTM / transfer-under-budget ---
    ltm_enabled: bool = True
    wm_budget_ms: float = 50.0  # memory-management-latency setpoint (matches
        # gate G2's "p95 update latency tracks the setpoint without
        # oscillation"): enforce_budget() transfers ONE victim group per
        # call when the last measured update took longer than this, and
        # does nothing otherwise -- see memory.py's controller note for
        # why single-victim-per-call is the deliberate choice.
        # WP-J2 (Orin port): this is compared against mem_duration_ms, the
        # time spent in memory.add + retrieval + verify + graph.optimize
        # on a keyframe frame (0.0 on non-keyframe frames) -- NOT total
        # frame time. Feature extraction and odometry don't scale with WM
        # size, so they must not drive eviction; a platform with a
        # proportionally slower/faster frontend than the reference machine
        # (e.g. porting to different hardware) would otherwise evict at a
        # different rate for reasons unrelated to actual memory pressure,
        # silently changing loop-closure recall. Retune this setpoint per
        # platform from the platform's own measured mem_duration_ms
        # distribution, not from total per-frame latency.
    wm_min_resident: int = 20  # floor: enforce_budget() never transfers
        # a node created within the most recent wm_min_resident keyframes,
        # regardless of budget pressure -- this is what keeps the
        # immediate spatial neighbourhood of "now" always retrieval-ready
        # without a separate radius/recency heuristic.
    wm_rehearsal_similarity: float = 0.6  # moved out of memory.py's
        # hardcoded default so it's part of the hashed, versioned config
        # like every other tunable in this file (Phase-0 left it as a
        # bare class attribute; no behaviour change, same 0.6 value).

    imagery_cache_enabled: bool = True  # WP-K3 (Phase A3): when a node is
        # transferred WM -> LTM, Memory drops its full-resolution RGB and
        # depth from RAM (frozen contract: Signature.rgb/depth are "kept
        # only while in WM", and LTM's SQLite blob never stores them --
        # both unchanged). Before this flag, that ALSO meant those nodes
        # vanished from map.ply entirely, because mapping/cloud.py's
        # assemble_cloud() skips any node whose rgb/depth are None: on
        # corridor_v2 (184 keyframes, WM peaking ~52) roughly two thirds
        # of the keyframes contributed nothing to the exported map. With
        # this True, Memory writes each evicted node's rgb+depth to a
        # side-cache of lossless PNGs (a directory OUTSIDE the LTM
        # database -- retrieval never sees imagery again, only the
        # map exporter does, streaming one node at a time). False restores
        # the exact pre-WP-K3 behaviour (evicted nodes are not mapped).

    # --- vocabulary / retrieval ---
    vocab_size: int = 1000
    bow_top_k: int = 20  # NOT YET WIRED: Phase 0 scores retrieval by direct
        # descriptor matching (vpr/raw_match.py), not through the BoW index
        # this would configure. bow_index is still populated every keyframe
        # for future phases; see raw_match.py's own docstring. Not a bug --
        # documented Phase-0 scope, kept here as the intended P3 knob.

    # --- Phase 3 (WP-P3): retrieval upgrade ---
    retrieval_backend: str = "raw"  # "raw" (Phase 0's direct-Hamming
        # matching, still the default -- see raw_match.py's own findings
        # on why a fixed 1k vocabulary erased the discriminative signal
        # at Phase-0 WM scale) | "bow_incremental" | "learned" | "both".
        # Mirrors odometry_backend's own pattern: opt-in, flip only on a
        # gate-confirmed win, not silently defaulted to the new thing.
    p3_new_word_hamming_threshold: int = 70  # incremental vocabulary
        # (vpr/incremental_bow.py): a descriptor reuses an existing word
        # if its Hamming distance to that word is <= this, out of 256
        # bits. EMPIRICALLY CALIBRATED, not a guess: on square6dof's
        # verified loop pair (node 136<->0), cross-frame Hamming
        # distance to the nearest same-point descriptor has percentiles
        # [5,10,25,50,75] = [24,28,38,51,62] (out of 256 bits) -- the
        # first version of this file defaulted to 40 (a "looks
        # conservative" guess) and it measurably broke recall: 0/1 loop
        # closures found vs raw_match's 1, because it only let ~29% of
        # a true match's descriptors reuse a word. 70 recovers all
        # loops raw_match finds (see WP_P3_Findings.md for the full
        # sweep, including why 80 is even better on THIS fixture but
        # left at the more conservative 70 pending a check against
        # aliasing_rooms's false-positive risk).
    p3_global_desc_dim: int = 256  # output dimensionality of the learned-
        # channel global descriptor (vpr/global_desc.py).
    p3_ann_backend: str = "auto"  # "auto" | "hnswlib" | "numpy" -- see
        # vpr/ann_index.py; mirrors backend_gtsam.py's own
        # optional-dependency-with-fallback pattern for the same
        # unresolved "can you pip install on the target machine"
        # question from the architecture doc's section 8.
    p3_ann_initial_capacity: int = 2000

    # --- Phase 4 (WP-P4): real information matrices ---
    use_hessian_info: bool = False  # opt-in, like odometry_backend/
        # retrieval_backend. When True, odometry.py, odometry_f2m.py,
        # AND verify.py all switch from their heuristic
        # eye(6)*inlier_ratio*K info matrices to pnp_info.py's real
        # PnP-Hessian derivation. Deliberately a single flag covering
        # ALL THREE sites together, not three independent ones -- WP-B2
        # tried wiring only the odometry side and found it made graph
        # optimisation "functionally inert" (a ~300,000x scale mismatch
        # against the still-heuristic loop-link side, see
        # WP_B2_Findings.md); flipping only one side again would
        # reproduce exactly that failure. See WP_P4_Findings.md for the
        # end-to-end numbers now that both sides are consistent.
    pnp_sigma_px: float = 1.25  # assumed per-axis pixel noise std for
        # every pnp_info_matrix* call. Reused directly from WP-B2's own
        # calibration (static_60s, NEES-based) rather than
        # re-calibrated separately for verify.py's loop-link context --
        # justified, not just convenient: both odometry.py's and
        # verify.py's PnP calls measure uncertainty in the SAME
        # underlying quantity (ORB keypoint pixel-localisation noise
        # under this renderer's noise model), which doesn't change with
        # baseline length or which frame pair is involved; what differs
        # (inlier count, point geometry) is already captured by the
        # Hessian itself, not by sigma_px. Flagged in
        # WP_P4_Findings.md as worth an independent direct check, not
        # asserted as beyond doubt.

    # --- bayes filter ---
    bayes_neighbor_spread: float = 0.8   # NOT YET WIRED: BayesFilter
        # (loop/bayes.py) hardcodes decay_per_step=0.15 as a constructor
        # default instead of reading this. F8 hygiene pass (WP-B4)
        # confirmed via grep that nothing reads this field; deliberately
        # left declared-but-inert rather than silently wired to a
        # DIFFERENT numeric default (0.8 vs the hardcoded 0.15), which
        # would change Bayes-filter dynamics and require the same
        # baseline re-validation as any other behaviour change -- not a
        # "hygiene" fix. Wire it up as its own gated change if pursued.
    bayes_new_place_prior: float = 0.6   # NOT YET WIRED: BayesFilter's
        # _ensure() hardcodes a fixed 0.5 alloc split instead of reading
        # this. Same caveat as bayes_neighbor_spread above.
    bayes_diffusion_enabled: bool = False  # WP-L2: graph-neighbour belief diffusion
        # in BayesFilter._predict. Off by default until re-baselined on the
        # target machine (it changes Bayes dynamics); see WP_L_Findings.md.
        # With it on, each step a candidate hands `bayes_diffusion_rate` of its
        # belief to its graph neighbours (odom/loop-adjacent nodes that are
        # themselves current candidates), split evenly -- so evidence gathered
        # on node X carries over to X's neighbours as the camera moves through
        # a revisited region, instead of every neighbour restarting from the
        # tiny 'new place' seed each frame. This is RTAB-Map's own mechanism;
        # bayes_neighbor_spread above stays declared-but-inert (its 0.8 default
        # was chosen for a different semantics and is deliberately not reused).
    bayes_diffusion_rate: float = 0.3    # WP-L2: fraction of a candidate's belief
        # handed to its neighbours per step (only when bayes_diffusion_enabled
        # and the candidate has at least one neighbour in the candidate set;
        # otherwise nothing moves). Total belief mass is conserved exactly.
    hypothesis_threshold: float = 0.15
    hypothesis_hysteresis: int = 2
    wm_max_nodes: int = 0  # WP-L5: DETERMINISTIC working-memory cap. 0 = off (the
        # frozen WP-P2 behaviour: eviction is driven only by wm_budget_ms, i.e. by
        # wall-clock time). >0: whenever WM holds more than this many nodes,
        # enforce_budget evicts one victim (per the wm_evict_policy) regardless of
        # timing. Why it exists: with the time-driven budget, WHICH nodes are in WM
        # at any moment depends on how fast the machine happened to run that
        # second, so two runs of the same fixture on the same machine evict
        # different nodes and can give different loop-closure results (observed on
        # corridor_lap13, WP_L_Findings.md). Set wm_budget_ms very large and
        # wm_max_nodes to a fixed size for a reproducible experiment; on hardware
        # it also lets you pick WM size directly instead of via a latency target.
    wm_evict_policy: str = "oldest"  # WP-L4: which WM node goes to LTM when the
        # budget controller asks for a victim. "oldest" (default, unchanged) = the
        # frozen WP-P2 rule: lowest rehearsal weight, oldest first -- and because
        # rehearsal weights are inert unless a vocabulary is loaded (all 1), that is
        # simply FIFO: the START of the run is always the first thing evicted. On a
        # ring/loop trajectory that is exactly the part the closing loop needs.
        # "redundancy" = evict the node whose neighbourhood is most crowded (most
        # other WM nodes within wm_redundancy_radius_m AND wm_redundancy_angle_deg of
        # viewing direction), ties -> lowest weight -> oldest. Under a fixed WM size
        # this thins the map to roughly uniform spatial coverage instead of keeping
        # only the recent stretch. Same one-victim-per-over-budget-call controller.
    wm_redundancy_radius_m: float = 0.5   # WP-L4
    wm_redundancy_angle_deg: float = 30.0  # WP-L4
    loop_min_path_m: float = 0.0   # WP-L1: a WM node is only a loop-closure CANDIDATE
        # if the ODOMETRY path travelled since it was created is at least this
        # many metres (cumulative keyframe-to-keyframe distance, so it is
        # independent of walking speed and of keyframe density). 0 = off (the
        # pre-WP-L behaviour). Why this exists: STM (10 keyframes) is the only
        # thing keeping the newest nodes out of the candidate set, and at
        # ~11 cm/keyframe that is ~1 m -- corridor_v2's 14 'loop closures'
        # were ALL 1.6-2.9 s apart (zero real), each just a node leaving STM
        # matching its own recent past. Such a link is geometrically correct
        # but carries no information (odometry already constrains those two
        # nodes to a few mm) and crowds real candidates out of the belief and
        # out of the z-score normalisation. The filter is applied BEFORE
        # scoring, so it also removes their cost from retrieval.
    loop_min_time_s: float = 0.0   # WP-L1: same, but by elapsed capture time.
        # Both apply (a node must satisfy each non-zero bound). Time alone is
        # a poor proxy (a camera standing still for 10 s has travelled nowhere);
        # prefer loop_min_path_m and use this only as a belt-and-braces bound.

    # --- verification ---
    verify_min_inliers: int = 20
    verify_min_inlier_ratio: float = 0.3
    verify_reproj_px: float = 3.0
    verify_bidirectional_trans_m: float = 0.05
    verify_bidirectional_rot_deg: float = 2.0
    verify_max_translation_m: float = 3.0    # sanity bound: a WM candidate is
        # already appearance-matched (i.e. plausibly nearby); a "verified"
        # link proposing a huge jump is much more likely to be a PnP
        # bas-relief/degenerate solution (common on near-planar scenes,
        # e.g. corridor walls) than a real distant-but-textured-alike loop.
        # Full ICP-based refinement (Phase 4) removes the need for this
        # blunt cap; until then it is cheap, safe insurance.

    # --- relocalization (single-snapshot, monocular query against a
    # frozen map pack -- see pyslam/loop/relocalize.py) ---
    reloc_shortlist_backend: str = "global_desc"  # "global_desc" (default,
        # exact cosine over vpr/global_desc.py's random-projection
        # embedding -- cost independent of map size) or "brute" (exact
        # Hamming match against every descriptor in the pack; fine up to
        # a few hundred nodes, grows linearly beyond that -- see
        # RELOCALIZATION.md's sizing note). No BoW backend here: the
        # incremental vocabulary's O(N*V) Hamming matrix is the thing
        # that made a 3+GB single allocation possible on an 8GB board at
        # map sizes this feature targets -- see WP-RELOC findings.
    reloc_shortlist_topk: int = 15
    reloc_min_inliers: int = 25          # above verify_min_inliers (20): a
        # monocular query is a weaker constraint than a verified two-way
        # loop closure, so it is held to a higher bar, not the same one.
    reloc_min_inlier_ratio: float = 0.35
    reloc_reproj_px: float = 3.0
    reloc_min_grid_cells: int = 6        # of grid_cols*grid_rows (default
        # 48): inliers must span at least this many distinct feature-grid
        # cells, or the candidate is rejected -- a spatial coverage floor
        # against inliers clustered on one small image region (a single
        # poster or object), NOT a 3D-planarity test: a fronto-parallel
        # wall filling the whole frame can pass this easily, since
        # perspective gives real depth variation across the image even
        # for a flat surface (confirmed empirically against the
        # synthetic renderer -- see RELOCALIZATION.md). Genuine
        # ill-conditioning (bas-relief-style ambiguity) is caught by the
        # covariance gate below, which looks at the actual PnP Hessian.
    reloc_consensus_trans_m: float = 0.25   # two independently-verified
        # candidates' world poses must agree within this translation ...
    reloc_consensus_rot_deg: float = 5.0    # ... and this rotation, or the
        # query is AMBIGUOUS rather than LOCALIZED. Monocular stand-in for
        # verify.py's own bidirectional PnP check, which a depthless query
        # cannot run (direction 1 there needs the QUERY's own 3D points).
    reloc_single_candidate_min_inliers: int = 40   # sparse-map fallback:
        # when only ONE candidate verifies (expected on maps around the
        # 80-keyframe floor, where a query viewpoint may genuinely
        # overlap only one keyframe), it is accepted alone under a
        # stricter bar instead of being discarded for lack of a second
        # opinion. Higher than reloc_min_inliers.
    reloc_single_candidate_min_grid_cells: int = 10
    reloc_single_candidate_max_sigma_trans_m: float = 0.25  # tighter than
        # reloc_max_sigma_trans_m below, for the same single-candidate case.
    reloc_max_sigma_trans_m: float = 0.5   # translation-covariance gate
        # (from pnp_info_matrix's own Hessian, not a heuristic): a
        # well-conditioned-LOOKING solution that is actually poorly
        # constrained (e.g. a long, narrow inlier cluster) fails here
        # even if inlier count/ratio/RMS all looked fine.
    reloc_retrieval_margin: float = 1.3    # best candidate's shortlist
        # likelihood must exceed the runner-up's by this factor UNLESS
        # the two are graph neighbours (adjacency.json) -- a near-tie
        # between neighbouring keyframes is the healthy, expected case;
        # a near-tie between unrelated places is perceptual aliasing.
    reloc_blur_reject_var: float = 40.0    # variance-of-Laplacian floor
        # for the burst-capture sharpest-frame selector (relocalize.py);
        # a burst where EVERY frame is below this is rejected outright
        # before any feature work -- motion blur is the single most
        # common cause of a spurious NOT_FOUND on live hardware.

    # --- graph ---
    prior_sigma_pos: float = 1e-6  # NOT YET WIRED: no code ever creates a
        # Link with kind="prior" (confirmed via grep) -- the frozen
        # contract (architecture doc section 3) reserves this Link kind
        # for a later phase (e.g. IMU-anchored or GPS-anchored priors).
        # Declared here ahead of that need, not currently consumed.
    prior_sigma_rot: float = 1e-6  # see prior_sigma_pos.
    odom_sigma_pos: float = 0.02  # NOT YET WIRED: odom link info matrices
        # are set directly in frontend/odometry.py and odometry_f2m.py as
        # eye(6) * inlier_ratio * 100 (the heuristic WP-B2 exists to
        # replace with something derived from the PnP Hessian). Same
        # "declared but inert, don't silently wire to a different value"
        # reasoning as the bayes_* fields above -- fold this into WP-B2's
        # own gated change, not this hygiene pass.
    odom_sigma_rot_deg: float = 1.0  # see odom_sigma_pos.
    loop_huber_delta: float = 3.5   # ~sqrt(chi2.ppf(0.95, df=6))=3.55, the
        # whitened-residual VECTOR NORM at which a loop factor starts being
        # down-weighted. Phase 0 used 1.0, which (since the whitened
        # residual in backend_native._factor_residual already has ~unit
        # variance per component once covariances are correctly
        # calibrated -- WP-B2) suppresses most genuine loops, not just
        # outliers: 1.0 is roughly a 1-sigma-per-component bound, whereas
        # a correct 6-DoF link at the 95th percentile of its own noise
        # model has whitened norm ~3.5. See Phase 1 plan finding F6b.
    loop_accept_chi2_995: float = 18.548  # scipy.stats.chi2.ppf(0.995, df=6);
        # post-optimisation loop-acceptance bound (WP-B4): after adding a
        # new loop link and re-optimising, if that link's own NEES
        # (err^T @ info @ err against the OPTIMISED poses) exceeds this,
        # the optimisation is rolled back and the loop is rejected rather
        # than accepted into a map it makes worse.

    # --- Phase 4 (WP-P4): robust back-end ---
    loop_robust_kernel: str = "huber"  # "huber" (default, unchanged) | "dcs"
        # (Dynamic Covariance Scaling -- backend_native.py's
        # _factor_residual). Opt-in: gate G4 measures whether DCS is
        # actually a clear win on injected-wrong-loop robustness before
        # this default would ever change, same pattern as every other
        # backend/kernel choice in this file.
    dcs_xi: float = 6.0  # DCS's "Xi" parameter -- set to the loop
        # factor's DOF (6), the expected chi-square value for a
        # genuinely correct factor. See backend_native.py's
        # _factor_residual for the formula and rationale.
    proximity_enabled: bool = False  # opt-in, same pattern as every
        # other P2-P4 backend choice in this file. When True, every
        # keyframe also attempts geometry-only proximity detection
        # (proximity/detect.py) alongside appearance-triggered loop
        # closure.
    proximity_radius_m: float = 0.5  # candidate pairs: WM nodes whose
        # CURRENT pose_map positions are within this distance.
    proximity_min_index_gap: int = 15  # excludes trivially-adjacent
        # keyframes -- same constant and same purpose as gate G3's own
        # true-pair definition (tests/gates/test_g3.py).

    # --- Phase 5 (WP-P5.2): loose IMU coupling -- gravity/tilt prior ---
    gravity_prior_enabled: bool = False  # opt-in, same pattern as every
        # other P2-P4 backend choice in this file. When True, a
        # tilt-only Link(kind="prior") is attempted at every keyframe
        # once a gravity-aligned world frame has been established (see
        # pyslam/imu/gravity.py) -- silently a no-op for any keyframe
        # whose own IMU window isn't quasi-static enough to trust (see
        # gravity.is_quasi_static's own conservative-by-design
        # docstring), so this flag alone does not guarantee any prior
        # actually fires on a given run.
    gravity_prior_sigma_tilt_rad: float = 0.035  # ~2 degrees. Info
        # weight on the tilt (roll/pitch) component of the prior is
        # 1/this^2; NOT independently calibrated against real/synthetic
        # accelerometer noise the way pnp_sigma_px was (WP-B2's NEES
        # calibration) -- a reasonable-looking default, flagged as an
        # open calibration item, not asserted as correct.
    gravity_prior_gyro_thresh_rad_s: float = 0.08  # see
        # gravity.is_quasi_static's own docstring for the reasoning.
    gravity_prior_accel_std_thresh_mps2: float = 0.5  # see
        # gravity.is_quasi_static's own docstring.

    # --- WP-T0: LOST-recovery bridge link ---
    bridge_link_info_scale: float = 1e-3  # WP-T0: when tracking goes
        # LOST and force_new_keyframe() starts a fresh reference frame,
        # pipeline.py used to add NO odometry link between the last
        # connected keyframe and the new one (last_keyframe_link_T was
        # None). That leaves the new node -- and every node after it,
        # until the next successful link -- with NO factor touching it
        # at all if no loop closure ever finds it, or as an unanchored
        # second connected component if one does. GTSAM's LM optimizer
        # then either can't build a full elimination ordering (a node
        # with zero factors) or solves an under-constrained second
        # component (singular/free-floating) -- this is the exact
        # 'inconsistent arguments' RuntimeError seen on corridor_v2's
        # hardware run. The fix: always add a Link(kind='bridge',
        # T_ab=identity) across a LOST gap, weighted by this constant
        # (very low information = 'we assume no motion happened, but
        # trust this hardly at all') so the graph stays one connected,
        # fully-observed component. A real loop closure across the gap
        # will dominate this weak prior once verified. See
        # pipeline.py's _run_loop and Link.kind=='bridge' handling in
        # both graph backends.

    # --- WP-M: gyro/constant-velocity LOST-recovery bridge ---
    bridge_mode: str = "identity"  # "identity" (Phase 0/WP-T0, unchanged
        # default) | "gyro" (WP-M). opt-in, same pattern as every other
        # backend/kernel choice in this file. When "gyro", pipeline.py
        # replaces the identity bridge above with pyslam/imu/bridge.py's
        # per-component estimate: gyro-integrated rotation and constant-
        # velocity-extrapolated translation, each falling back to the
        # Phase-0 identity value independently if its own inputs aren't
        # trustworthy enough (see that module's docstring). This is the
        # item both WP_K_Findings.md (section 2: measured ~7.7deg
        # rotation error and essentially all of the missing path length
        # per LOST event) and WP_L_Findings.md (section 7) flagged as
        # the single biggest remaining lever -- ahead of the IMU tilt
        # prior.
    bridge_gyro_min_samples: int = 5  # below this many IMU rows in the
        # gap window, the gyro integration is not trusted (too few
        # samples to average out noise) and rotation falls back to
        # identity -- same conservative-fallback philosophy as
        # gravity.is_quasi_static's own threshold.
    bridge_rotation_sigma_rad: float = 0.05  # ~2.9 degrees. Info weight
        # on the gyro-derived rotation block is 1/this^2 -- a
        # reasonable-looking default from the synthetic fixtures this
        # shipped with (WP_M_Findings.md), NOT independently calibrated
        # against real gyro noise the way pnp_sigma_px was (WP-B2's NEES
        # calibration); flagged as an open calibration item, same status
        # gravity_prior_sigma_tilt_rad already carries.
    bridge_velocity_sigma_base_mps: float = 0.3  # translation info
        # weight is 1/sigma^2 where sigma = this + _growth * dt -- the
        # base term for a near-zero-length gap (still not perfectly
        # trusted: the pre-LOST velocity estimate is itself only a
        # 2-sample finite difference).
    bridge_velocity_sigma_growth_mps_per_s: float = 1.0  # additional
        # position sigma PER SECOND of elapsed gap time -- see
        # pyslam/imu/bridge.py's module docstring on why translation
        # (unlike rotation) is deliberately trusted LESS the longer the
        # LOST gap runs.

    # --- WP-LIVE N2: local RGB-D relocalization on LOST, tried BEFORE
    # the bridge-link fallback above. Opt-in, off by default, same
    # discipline every other opt-in in this file uses -- see
    # pyslam/loop/local_reloc.py's module docstring for why this is a
    # strictly better-conditioned problem than the frozen-map
    # relocalizer's 2D-3D PnP (both sides here carry real depth).
    local_reloc_enabled: bool = False
    local_reloc_max_candidates: int = 8  # how many recent working-memory
        # keyframes to try, most-recently-added first, before giving up
        # and falling back to bridge_mode
    local_reloc_min_inliers: int = 25
    local_reloc_min_inlier_ratio: float = 0.35
    local_reloc_max_rms_m: float = 0.03
    local_reloc_dist_thresh_m: float = 0.05  # RANSAC inlier distance for the 3D-3D solve

    # --- WP-T3: gravity-aligned world frame for trajectory export ---
    gravity_align_hold_s: float = 1.5  # seconds of IMU data from the
        # START of a run collected into PipelineResult.startup_imu_window,
        # intended to be a static hold so pyslam.core.gravity_frame can
        # estimate an up direction independent of (and before) any
        # gravity_prior_enabled tilt-prior machinery. Purely a trajectory-
        # export concern -- does not feed into the graph itself.

    # (F8 hygiene, WP-B4: `seed` and `log_level` fields were removed here.
    # Neither was ever read -- every seed is threaded explicitly through
    # function args (SyntheticSource(seed=...), build(seed=...), etc.),
    # and get_logger() takes its own local `level` default, never this
    # Config. Confirmed via grep that no call site anywhere constructs
    # Config(seed=...) or Config(log_level=...), so removing them changes
    # no behaviour. If per-run log verbosity is wanted later, thread it
    # through get_logger's own `level` param instead of resurrecting a
    # Config field nothing reads.)

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"config": self.to_dict(), "hash": self.hash()}, f, indent=2)

    @staticmethod
    def load(path: str) -> "Config":
        with open(path) as f:
            d = json.load(f)
        return Config(**d["config"])


# ---------------------------------------------------------------------------
# WP-K2 (Phase A2): command-line config overrides
#
# Phase_Evaluation.md has documented a `--config-override key=value` flag
# for a long time, but no runner ever implemented it -- toggling
# odometry_backend / use_hessian_info / proximity_enabled meant editing
# this file's defaults by hand, which is error-prone (a forgotten edit
# silently contaminates the next A/B comparison) and leaves the run
# directory's config.json as the only record of what actually ran. This
# section is the one shared implementation every runner uses (run_slam.py,
# run_bag.py, run_synth.py, tools/phase_a_baseline.py).
#
# Values are coerced from the FIELD'S OWN declared type (so `--config-override
# use_hessian_info=true` yields a real bool, not the truthy string "false"),
# and enumerated string fields are validated against the allowed set so a
# typo like retrieval_backend=incremental (the real value is
# "bow_incremental" -- a mistake Phase_Evaluation.md itself made) fails
# loudly with the valid choices instead of silently falling through to a
# different code path.
# ---------------------------------------------------------------------------
from dataclasses import fields as _dc_fields, replace as _dc_replace  # noqa: E402
import difflib as _difflib  # noqa: E402

# Only fields whose valid values are a closed set. Keep in sync with the
# docstrings above; a gate (tests/gates/test_ga.py) checks every entry
# here is actually a Config field and that the DEFAULT is in its own set.
CONFIG_CHOICES: dict = {
    "odometry_backend": ("f2f", "f2m"),
    "retrieval_backend": ("raw", "bow_incremental", "learned", "both"),
    "loop_robust_kernel": ("huber", "dcs"),
    "p3_ann_backend": ("auto", "hnswlib", "numpy"),
    "wm_evict_policy": ("oldest", "redundancy"),
    "bridge_mode": ("identity", "gyro"),
    "reloc_shortlist_backend": ("global_desc", "brute"),
}

_TRUE = {"1", "true", "yes", "on", "y", "t"}
_FALSE = {"0", "false", "no", "off", "n", "f"}


def _coerce(field_name: str, type_name: str, raw: str):
    t = type_name.replace("typing.", "").strip()
    if t == "bool":
        v = raw.strip().lower()
        if v in _TRUE:
            return True
        if v in _FALSE:
            return False
        raise ValueError(f"{field_name}: expected a boolean (true/false), got {raw!r}")
    if t == "int":
        try:
            return int(raw)
        except ValueError:
            # allow "20.0" for an int field, but not "20.5" (silently truncating is a bug)
            f = float(raw)
            if f != int(f):
                raise ValueError(f"{field_name}: expected an integer, got {raw!r}")
            return int(f)
    if t == "float":
        return float(raw)
    if t == "str":
        return raw
    raise ValueError(f"{field_name}: unsupported field type {type_name!r} for command-line override")


def parse_override(item: str):
    """'key=value' -> (key, coerced_value). Raises ValueError with a
    human-readable message (including close-match suggestions for typos)."""
    if "=" not in item:
        raise ValueError(f"bad --config-override {item!r}: expected key=value")
    key, raw = item.split("=", 1)
    key = key.strip()
    by_name = {f.name: f for f in _dc_fields(Config)}
    if key not in by_name:
        close = _difflib.get_close_matches(key, list(by_name), n=3, cutoff=0.6)
        hint = f" -- did you mean {', '.join(close)}?" if close else ""
        raise ValueError(f"unknown Config field {key!r}{hint}")
    value = _coerce(key, str(by_name[key].type), raw)
    if key in CONFIG_CHOICES and value not in CONFIG_CHOICES[key]:
        raise ValueError(f"{key}={value!r} is not valid; choose one of {list(CONFIG_CHOICES[key])}")
    return key, value


def apply_overrides(cfg: Config, overrides) -> Config:
    """Return a NEW Config with every 'key=value' string in `overrides`
    applied (Config is frozen). None/empty -> cfg unchanged (same object)."""
    if not overrides:
        return cfg
    kw = {}
    for item in overrides:
        k, v = parse_override(item)
        kw[k] = v
    return _dc_replace(cfg, **kw)


def add_config_args(ap) -> None:
    """Attach the shared --config-override flag to an argparse parser."""
    ap.add_argument("--config-override", dest="config_override", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="override a Config field for this run, e.g. "
                         "--config-override proximity_enabled=true "
                         "--config-override odometry_backend=f2m (repeatable; "
                         "the effective config is saved to the run directory's config.json)")


def config_from_args(args) -> Config:
    """Build the run's Config from parsed args, exiting with a clean
    message (not a traceback) on a bad override."""
    try:
        return apply_overrides(Config(), getattr(args, "config_override", None))
    except ValueError as e:
        raise SystemExit(f"error: {e}")
