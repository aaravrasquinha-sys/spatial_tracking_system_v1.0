# M_corr_rot4x

overrides: `['proximity_enabled=true', 'bridge_mode=gyro', 'bridge_rotation_sigma_rad=0.20']`  commit: `64ae041`  backend: `auto`

median over seeds

| scenario    | kf  | LOST | loops | real | trivial | wrong | revisit_opp | recall | prox | prox_wrong | aATE_odom | aATE_online | aATE_final | ATE_final | rot_deg/m | trans_% | vert_rms_m | horiz_rms_m | bridge_lie_m | path_gt_m | path_est_m | map_kf | map_kf_old | mem_p95_ms | fps | wm_max | kf_by_inl | kf_by_tr | kf_by_rot | kf/m | wm_oldest |
|-------------|-----|------|-------|------|---------|-------|-------------|--------|------|------------|-----------|-------------|------------|-----------|-----------|---------|------------|-------------|--------------|-----------|------------|--------|------------|------------|-----|--------|-----------|----------|-----------|------|-----------|
| corridor_v2 | 116 | 9    | 11    | 6    | 5       | 0     | 0           | -      | 3    | 0          | 122.6     | 96.9        | 96.9       | 52.4      | 0.712     | 1.87    | 0.123      | 0.961       | 2.37         | 15.6      | 13.3       | 116    | 40         | 7630       | 2.3 | 30     | 69        | 45       | 6         | 7.4  | 191       |
