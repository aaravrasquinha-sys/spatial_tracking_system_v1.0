# L_corr_b1b2red

overrides: `['proximity_enabled=true', 'loop_min_path_m=1.5', 'bayes_diffusion_enabled=true', 'wm_evict_policy=redundancy']`  commit: `0cc256f`  backend: `native`

median over seeds

| scenario    | kf  | LOST | loops | real | trivial | wrong | revisit_opp | recall | prox | prox_wrong | aATE_odom | aATE_online | aATE_final | ATE_final | rot_deg/m | trans_% | vert_rms_m | horiz_rms_m | bridge_lie_m | path_gt_m | path_est_m | map_kf | map_kf_old | mem_p95_ms | fps | wm_max | kf_by_inl | kf_by_tr | kf_by_rot | kf/m | wm_oldest |
|-------------|-----|------|-------|------|---------|-------|-------------|--------|------|------------|-----------|-------------|------------|-----------|-----------|---------|------------|-------------|--------------|-----------|------------|--------|------------|------------|-----|--------|-----------|----------|-----------|------|-----------|
| corridor_v2 | 179 | 15   | 0     | 0    | 0       | 0     | 1           | 0.00   | 5    | 0          | 150.9     | 150.1       | 150.1      | 64.9      | 0.707     | 1.63    | 0.940      | 1.170       | 3.84         | 23.8      | 20.0       | -      | -          | 366        | 8.0 | 38     | 102       | 69       | 15        | 7.5  | 0         |
