# corridor_probe

overrides: `['proximity_enabled=true']`  commit: `ec3c91f`  backend: `native`

median over seeds

| scenario    | kf  | LOST | loops | real | trivial | wrong | prox | prox_wrong | aATE_odom | aATE_online | aATE_final | ATE_final | rot_deg/m | trans_% | vert_rms_m | horiz_rms_m | bridge_lie_m | path_gt_m | path_est_m | map_kf | map_kf_old | mem_p95_ms | fps | wm_max |
|-------------|-----|------|-------|------|---------|-------|------|------------|-----------|-------------|------------|-----------|-----------|---------|------------|-------------|--------------|-----------|------------|--------|------------|------------|-----|--------|
| corridor_v2 | 179 | 15   | 14    | 0    | 14      | 0     | 5    | 0          | 150.9     | 148.6       | 148.6      | 65.4      | 0.707     | 1.63    | 0.917      | 1.170       | 3.84         | 23.8      | 20.0       | 179    | 42         | 13499      | 1.3 | 32     |
