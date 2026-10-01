# L_lap500_red

overrides: `['proximity_enabled=true', 'wm_budget_ms=1000000', 'wm_max_nodes=45', 'loop_min_path_m=3.0', 'wm_evict_policy=redundancy']`  commit: `0cc256f`  backend: `native`

median over seeds

| scenario       | kf  | LOST | loops | real | trivial | wrong | revisit_opp | recall | prox | prox_wrong | aATE_odom | aATE_online | aATE_final | ATE_final | rot_deg/m | trans_% | vert_rms_m | horiz_rms_m | bridge_lie_m | path_gt_m | path_est_m | map_kf | map_kf_old | mem_p95_ms | fps | wm_max | kf_by_inl | kf_by_tr | kf_by_rot | kf/m | wm_oldest |
|----------------|-----|------|-------|------|---------|-------|-------------|--------|------|------------|-----------|-------------|------------|-----------|-----------|---------|------------|-------------|--------------|-----------|------------|--------|------------|------------|-----|--------|-----------|----------|-----------|------|-----------|
| corridor_lap13 | 187 | 16   | 4     | 4    | 0       | 0     | 9           | 0.44   | 8    | 0          | 157.8     | 134.3       | -          | -         | 0.784     | 1.59    | -          | -           | 4.22         | 25.2      | -          | -      | -          | 9224       | 1.6 | 47     | 107       | 73       | 16        | 7.4  | 4         |
