# baselines/

Output of `python3 -m pyslam.tools.phase_a_baseline` (WP-K5). One `<label>.json` (full per-run detail +
per-scenario aggregate) and `<label>.md` (median table) per invocation, rewritten after every run.

Files committed here are REFERENCE POINTS FROM THE x86 DEV SANDBOX (1 core, native pose-graph backend,
`proximity_enabled=true`), not Orin numbers -- their accuracy columns are comparable to Orin results run with
the same settings, their timing columns are not:

* `smoke_sq`        square6dof seed 1 -- tool smoke test; anchored ATE after finalize 0.74 cm matches the
                    post-mirror-fix number recorded in WP_T_Findings.md.
* `corridor_probe`  corridor_v2 seed 1 -- the ground-truth audit of the fixture that struggled (see
                    WP_K_Findings.md section 2).

Do not overwrite `pyslam/tools/baseline_store.json` (the frozen WP-A3 gate) with anything from here.
Your own Orin baselines: `python3 -m pyslam.tools.phase_a_baseline --label A5_base --config-override proximity_enabled=true`

WP-L additions (all sandbox, x86, native backend, seed 1):
* `L_small_{base,b1,b1b2}`  square6dof/room_orbit/aliasing_rooms: no gap / gap 1.5 m / gap + diffusion.
* `L_corr_b1b2red`          corridor_v2: B1+B2+redundancy (only 1 keyframe has a true revisit partner).
* `L_lap_base`              corridor_lap13 default behaviour: 20 loops, none closes the lap.
* `L_lapD_fifo_gap`         corridor_lap13 full 624 frames, FIFO + cap 45 + 3 m gap: 0 loops.
* `L_lap500_{red,fifo}`     corridor_lap13 first 500 frames, cap 45 + 3 m gap: the audited redundancy-vs-FIFO comparison
                            in WP_L_Findings.md section 3.2 (`--no-finalize`, so ATE columns are online poses).

WP-M additions (sandbox, x86, native backend, seed 1, proximity_enabled=true):
* `M_corr_{identity,gyro}`  corridor_v2, first 300 of 480 frames: bridge_mode=identity vs
                            bridge_mode=gyro, the comparison in WP_M_Findings.md section 3.3.
                            Online ATE and vertical error improve substantially (x0.81, x0.16);
                            post-finalize ATE gets WORSE (x1.24).
* `M_corr_{rot2x,rot4x,vel3x,nearzero}`  same fixture, bridge_mode=gyro, at 4 different info-
                            matrix weights (up to and including matching identity's own weight
                            exactly). All 4 reproduce aATE_final=52.4cm bit-identically to the
                            default -- this FALSIFIES the original overconfidence hypothesis;
                            see WP_M_Findings.md section 3.3.1 for the corrected explanation and
                            what's still open.
