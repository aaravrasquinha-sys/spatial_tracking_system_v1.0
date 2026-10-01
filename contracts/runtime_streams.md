# Runtime streams

## poi.v1 (M5 -> viewers), one WebSocket `/ws`, plus HTTP `GET /api/scene`

| Message | Rate | Schema | Notes |
|---|---|---|---|
| `tracks` | per processed frame (~30 Hz) | `schemas/poi_tracks` | **always sent, even when `tracks` is empty**, so "nobody here" differs from "stream stalled" |
| `event` | when it happens | `schemas/poi_event` | `track_start/confirmed/lost/end`, `id_switch`, `calib_suspect` |
| `health` | 1 Hz | `schemas/poi_health` | `calib` is `ok / suspect / missing`; `clock` is `synced / device` |
| `scene` | once, HTTP | `schemas/poi_scene` | frame name, map_id, camera pose, intrinsics, asset URLs, walkable grid |
| `ping`/`pong` | client-driven | -- | clock offset for the VR latency measurement |

`frame` is `"local"` (Phase A, provisional) or `"room"` (Phase B, registered). Viewers gate on it.

### Behaviour added by `sts` (no schema change)

* **`calib_suspect` events carry `id = -1`** (they are about the system, not a track). The stock dashboard prints it as
  `calib suspect #-1` -- cosmetic (docs/OPEN_ITEMS.md #V1).
* While calibration is `suspect` and `watchdog.on_suspect == "suppress"` (default) `tracks` messages keep flowing with an
  **empty `tracks` array** and `health.calib = "suspect"`; with `"flag"` tracks keep flowing and only health says so.
  A viewer that ignores `health.calib` will therefore show an empty room during suspect state. A dedicated 2D-detections
  message is a reserved extension, not built.
* `map_id` on `tracks` is the real map id (the stock `run_present.py` sends `null`).

## Logs (JSONL, under `data/logs/<site>/<cam>/`)

| File | Writer | One line per |
|---|---|---|
| `detections/<cam>_<ts>.jsonl` | M3 | frame: every `Detection2D` (even empty frames) |
| `world_tracks/<cam>_<ts>.jsonl` | M4 | frame: `{t_capture, frame_id, phase, map_id, tracks[]}`; rotates at 200 MB |
| `events.jsonl` | `sts` | run start, calibration-state transition |
| `soak.jsonl` | `sts` | 10 s: rss, cpu, threads, frames, drops, temperature, watchdog counters |
| `run_<ts>.manifest.json` | `sts` | run: versions, git, config hashes, calibration sha, engine manifest, modified legacy files |

Track logs are a record of people's movements in your home: `sts retention` prunes them (defaults 14 d / clips 30 d / bags 7 d).
