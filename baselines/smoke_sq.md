# smoke_sq

overrides: `['proximity_enabled=true']`  commit: `ec3c91f`  backend: `native`

median over seeds

| scenario   | kf | LOST | loops | real | trivial | wrong | prox | prox_wrong | aATE_odom | aATE_online | aATE_final | ATE_final | rot_deg/m | trans_% | vert_rms_m | horiz_rms_m | bridge_lie_m | path_gt_m | path_est_m | map_kf | map_kf_old | mem_p95_ms | fps  | wm_max |
|------------|----|------|-------|------|---------|-------|------|------------|-----------|-------------|------------|-----------|-----------|---------|------------|-------------|--------------|-----------|------------|--------|------------|------------|------|--------|
| square6dof | 28 | 0    | 1     | 1    | 0       | 0     | 6    | 0          | 1.5       | 0.7         | 0.7        | 0.5       | 0.385     | 0.99    | 0.006      | 0.004       | 0.00         | 3.1       | 3.1        | 28     | 28         | 845        | 13.0 | 18     |
