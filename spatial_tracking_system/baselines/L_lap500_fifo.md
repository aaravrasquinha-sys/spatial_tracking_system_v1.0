# L_lap500_fifo

overrides: `['proximity_enabled=true', 'wm_budget_ms=1000000', 'wm_max_nodes=45', 'loop_min_path_m=3.0']`  commit: `0cc256f`  backend: `native`

median over seeds

| scenario       | kf  | LOST | loops | real | trivial | wrong | revisit_opp | recall | prox | prox_wrong | aATE_odom | aATE_online | aATE_final | ATE_final | rot_deg/m | trans_% | vert_rms_m | horiz_rms_m | bridge_lie_m | path_gt_m | path_est_m | map_kf | map_kf_old | mem_p95_ms | fps | wm_max | kf_by_inl | kf_by_tr | kf_by_rot | kf/m | wm_oldest |
|----------------|-----|------|-------|------|---------|-------|-------------|--------|------|------------|-----------|-------------|------------|-----------|-----------|---------|------------|-------------|--------------|-----------|------------|--------|------------|------------|-----|--------|-----------|----------|-----------|------|-----------|
| corridor_lap13 | 187 | 16   | 0     | 0    | 0       | 0     | 9           | 0.00   | 8    | 0          | 157.8     | 157.0       | -          | -         | 0.784     | 1.59    | -          | -           | 4.22         | 25.2      | -          | -      | -          | 355        | 2.7 | 45     | 107       | 73       | 16        | 7.4  | 359       |
