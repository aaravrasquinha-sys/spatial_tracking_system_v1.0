# poi_present — M5 (presentation: publisher + dashboard + WebXR viewer)

**May import** `poi_localization` / `poi_perception` record types. Never `pyslam` or `sts`. Does no 3D maths.

* **Consumes:** M4's `on_tracks(frame, records)` callback (live), a recorded world-track JSONL (replay), or a generator (synthetic).
* **Produces:** the `poi.v1` WebSocket feed + HTTP (`/`, `/dashboard/`, `/vr/`, `/api/scene`, `/map/...`, `/healthz`)
  on **one port, one dependency** (`websockets`). Contract: [`../contracts/runtime_streams.md`](../contracts/runtime_streams.md).
* **Server:** per-client bounded queues (tracks drop-oldest, events never dropped, slow client disconnected), token auth
  (`?token=`), optional TLS, health tick (fps, latency from `t_capture`, drops, `/sys/class/thermal`, clock sanity).
* **Web:** `web/dashboard/` (2D radar: walkable grid, tracks + trails + uncertainty ellipses, camera frustum, events, health)
  and `web/vr/` (Three.js WebXR: life-size with teleport, or ~1:20 dollhouse; renders ~100 ms in the past and interpolates on
  `t_capture`). Three.js and fonts are vendored; there is no build step.
* **The one edit STS made here** (`server/app.py`, additive, default-off): `PresentServer.calib_provider`
  (`.state()` -> `ok|suspect|missing`; an exception reads as `suspect`) and `publish_event_threadsafe(msg)`.
* **Entry point:** `scripts/run_present.py --source live|replay|synthetic`; the production path is `sts run`.
* **Tests:** `tests/unit/test_poi_present/` (33) + `tests/integration/test_present_hook.py`.

**Limits:** WebXR needs a secure context — the Quest will not enter VR from `http://<lan-ip>`; use `adb reverse` (dev) or HTTPS
with a cert the headset trusts (untethered) · the stock `run_present.py` sends `map_id: null` (sts sends the real one) ·
no `room.glb` (H3) · dashboard prints `calib_suspect` as `#-1` (V1) · bind to your LAN IP and set a token before leaving your desk.
