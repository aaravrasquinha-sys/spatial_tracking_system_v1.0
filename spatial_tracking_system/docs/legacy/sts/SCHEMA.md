# poi.v1 wire contract

This is the frozen contract between `poi_present`'s server and both
viewers (M5a dashboard, M5b WebXR). `poi_present/schema.py`,
`web/common/validate.js`, and every file in `docs/schema_examples/`
are all written against this document and against each other — a
pytest test (`tests/poi_present/test_schema_examples.py`) and a
manual check in the browser console both validate live traffic
against these same golden files, on purpose, so the three can't
quietly drift apart.

One WebSocket connection (`GET /ws`, or `wss://` with TLS configured)
carries three message types. One plain HTTP endpoint (`GET
/api/scene`) is a one-shot document, not part of the ~30 Hz stream.

## Connecting

```
ws://<host>:<port>/ws?token=<token>
```

`token` is required whenever `server.token` is set in the server
config (always, once you're past your own desk on loopback — see
README's Security section). Omitting the query param, or a
mismatched value, gets the HTTP `401` your browser's dev tools will
show as a failed upgrade.

## `tracks` — ~30 Hz, always sent, even empty

```json
{
  "type": "tracks",
  "schema": "poi.v1",
  "map_id": null,
  "cam_id": "cam0",
  "frame": "local",
  "seq": 18233,
  "t_capture": 1790061364.1234,
  "t_publish": 1790061364.1712,
  "tracks": [
    {
      "id": 7,
      "state": "confirmed",
      "p": [2.31, 1.08, 0.0],
      "v": [0.42, -0.05, 0.0],
      "cov_xy": [0.004, 0.0003, 0.0031],
      "height": 1.71,
      "conf": null,
      "src": "fused",
      "age_s": 12.4,
      "bbox": null
    }
  ]
}
```

- **`frame`** is `"local"` (Phase A — a provisional, camera-relative
  floor frame, no map) or `"room"` (Phase B — Module 2's calibration,
  registered to Module 1's map). This is the one field both viewers
  gate their entire rendering mode on. Phase A tracks are real
  positions relative to the *camera*, not fabricated data — they are
  provisional in the sense that they aren't yet registered to a room
  map, not in the sense of being fake.
- **`p`**/**`v`** are always 3-vectors, `z = 0` (the ground point).
- **`conf`** and **`bbox`** are `null` in Phase A (the underlying
  `WorldTrackPhaseA` record has no such fields) — never a fabricated
  value.
- An **empty `tracks` array** is a real, expected message ("nobody in
  frame right now"). It is not the same as no message at all — see
  "stream stalled" below.
- **`seq`** increments by 1 every frame this server has processed,
  regardless of how many clients are connected. A client noticing a
  gap in `seq` knows it dropped frames (its own queue was full, not a
  server problem — `event` messages are never dropped, only `tracks`).

## `event` — sent when it happens, never dropped

```json
{"type": "event", "event": "track_start", "id": 7, "t": 1790061364.10}
{"type": "event", "event": "track_confirmed", "id": 7, "t": 1790061364.20}
{"type": "event", "event": "track_lost", "id": 7, "t": 1790061376.40}
{"type": "event", "event": "track_end", "id": 7, "t": 1790061377.40}
```

`event` values: `track_start`, `track_confirmed`, `track_lost`,
`track_end`, `calib_suspect` (Module 6's watchdog, not wired up in
this build — `calib` in `health` is always `"missing"` in Phase A
until then), `id_switch` (optional, from M4's `2d_id_mismatch`
internal event — not emitted by this build).

`track_end` is the one event `poi_localization`'s `TrackManager`
never raises on its own (it just deletes the track silently after
`coasting_budget_s + lost_grace_s`) — `poi_present`'s
`adapter.EventDeriver` derives it by diffing the id set between
frames. See `poi_present/adapter.py`'s module docstring.

## `health` — 1 Hz

```json
{
  "type": "health",
  "fps": 29.8,
  "calib": "missing",
  "map_id": null,
  "temp_c": 54.2,
  "drop_rate": 0.0,
  "latency_ms": 41.3,
  "clients": 2,
  "clock": "synced"
}
```

- **`calib`**: `"ok" | "suspect" | "missing"`. `"missing"` whenever
  `frame == "local"` (no calibration exists yet). `"suspect"` is
  Module 6's watchdog, not implemented in this build.
- **`clock`**: `"synced" | "device"`. `"device"` means `t_capture`
  looks like device time, not epoch time (RealSense's
  `global_time_enabled` likely failed to apply — see
  `capture/realsense_source.py`) — both viewers hide latency readouts
  when they see this rather than showing a nonsense number.

## `pong` — reply to a client's `ping`

Client sends, over the same socket:

```json
{"type": "ping", "client_t": 1790061364.001}
```

Server replies:

```json
{"type": "pong", "client_t": 1790061364.001, "server_t": 1790061364.014}
```

The client keeps the sample with the lowest round-trip time and uses
`server_t - (client_t + rtt/2)` as its clock offset against the
server. This is how the WebXR viewer measures capture-to-photon
latency and times its 100 ms-behind interpolation window (see
README's M5b section).

## `GET /api/scene` — one-shot, not on the socket

```json
{
  "schema": "poi.v1",
  "frame": "local",
  "map_id": null,
  "cam_id": "cam0",
  "camera": {"R": [[1,0,0],[0,1,0],[0,0,1]], "t": [0.0, 0.0, 2.3], "height_m": 2.3},
  "intrinsics": {"fx": 606.75, "fy": 606.57, "cx": 320.19, "cy": 237.06, "width": 640, "height": 480},
  "fov_deg": [56.9, 41.2],
  "assets": [],
  "walkable": null,
  "server_time": 1790061364.0
}
```

Fetched once at connect and whenever a `tracks` message reports a
different `frame`/`map_id` than last seen (a daemon restart in the
other phase). `assets` is empty until Module 1 exists; once it does,
each entry is `{"kind": "mesh" | "points", "url": "/map/viewer/..."}`
served read-only from the locked map bundle. `walkable` is `null` in
Phase A (Section 8: the walkable gate does not exist in Phase A,
deliberately, so its absence on the wire matches its absence in the
pipeline).

## "Stream stalled" vs. "nobody here"

Because `tracks` is sent every processed frame with `tracks: []`
when no one is in frame, a viewer distinguishes:

- **Nobody in frame**: `tracks` messages keep arriving on schedule,
  each with an empty array.
- **Stream stalled**: no `tracks` message arrives for longer than a
  few frame periods. Both viewers watch the gap between `seq`
  values/arrival times and show a "STREAM STALLED" state distinct
  from "no one here" once it exceeds ~500 ms.
