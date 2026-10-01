# L_lap_base

overrides: `['proximity_enabled=true']`  commit: `0cc256f`  backend: `native`

median over seeds

| scenario       | kf  | LOST | loops | real | trivial | wrong | revisit_opp | recall | prox | prox_wrong | aATE_odom | aATE_online | aATE_final | ATE_final | rot_deg/m | trans_% | vert_rms_m | horiz_rms_m | bridge_lie_m | path_gt_m | path_est_m | map_kf | map_kf_old | mem_p95_ms | fps | wm_max | kf_by_inl | kf_by_tr | kf_by_rot | kf/m | wm_oldest |
|----------------|-----|------|-------|------|---------|-------|-------------|--------|------|------------|-----------|-------------|------------|-----------|-----------|---------|------------|-------------|--------------|-----------|------------|--------|------------|------------|-----|--------|-----------|----------|-----------|------|-----------|
| corridor_lap13 | 237 | 20   | 20    | 13   | 7       | 0     | 59          | 0.14   | 10   | 0          | 189.2     | 189.1       | 189.1      | 90.9      | 0.786     | 1.57    | 0.864      | 1.682       | 5.21         | 31.4      | 26.3       | -      | -          | 18628      | 1.2 | 47     | 138       | 89       | 19        | 7.5  | 484       |
