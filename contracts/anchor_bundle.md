# Anchor bundle (M2 -> M4, M5, watchdog)

One bundle **per camera**: `data/anchors/<site>/<cam_id>/`. It lives next to, never inside, the map bundle.

```
calibration.<cam>.json            ACCEPTED calibrations only. schemas/calibration.schema.json
calibration.<cam>.REJECTED.json   a rejected solve is written under THIS name; M4 cannot load it by accident
room_frame.json                   T_room_map (map frame -> room frame), floor fit + quadrants, walls (ids w0, w1, ...)
walkable.json                     {origin:[x,y], resolution, grid[row=y][col=x] of bool} in the ROOM frame
viewer/points.ply                 decimated map cloud IN THE ROOM FRAME (what M5 shows under the tracks)
<cam>_reference_depth.npz         depth_med + stable mask: the watchdog's reference
static_capture.npz, report.json   raw capture + every gate with its number and the config used
overlay.png depth_residual.png frustum_coverage.png top_down.png   visual sign-off
```

## Frames and conventions

* **room**: Z up, the fitted floor is z = 0, X/Y from the dominant wall direction (M2 derives it from the map alone).
  Units metres. `p_room = R p_cam + t` with `T_room_cam = [[R, t], [0, 1]]`.
* **cam**: the colour camera's optical frame (x right, y down, z forward); depth is aligned to colour.
* **M4 output** in room frame is `poi.v1` field `p` (ground point, z = 0); in Phase A it is `p_local`.
* The viewer applies one fixed rotation `(x, y, z)_room -> (x, z, -y)_three` at its scene root and nowhere else.

## What consumers require

| Consumer | Reads | Refuses when |
|---|---|---|
| M4 Phase B | `calibration` (`T_room_cam`, `sigma`, `map_id`), `walkable` | rotation not orthonormal / det < 0; `map_id != expected_map_id` |
| M5 | `viewer/points.ply` (+ `room.glb` if ever present), `walkable` | -- (serves what exists) |
| Watchdog | `calibration` (`gravity_up_cam`), `<cam>_reference_depth.npz`, `room_frame.json`, the map cloud | -- |
| `sts` consistency gate | everything above | see `sts/consistency.py` (22 checks, 23 with a pinned camera serial) |

`sigma{trans_m, rot_deg}` is what M4 adds to every position covariance. Better slightly pessimistic than overconfident.
