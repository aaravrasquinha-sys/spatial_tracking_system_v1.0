# poi_present — Module 5 (dashboard + WebXR)

This is the presentation layer: a 2D dashboard (M5a) and a WebXR
viewer (M5b) over Module 4's track stream, plus the publisher server
(`poi_present`) that turns M4 output into the `poi.v1` WebSocket
contract both viewers speak. See `SCHEMA.md` for that contract in
full.

It reads nothing but M4's already-computed positions — no SLAM, no
calibration math, no detection — so it lives alongside
`poi_perception` and `poi_localization` in this same repo without
having changed a line of either.

## Install

Already wired into this repo's `pyproject.toml` and
`requirements.txt` — `poi_present*` is in `packages.find`'s include
list, and `websockets`/`psutil` are in `requirements.txt`. From repo
root:

```bash
pip install -r requirements.txt
pip install -e .
pytest tests/test_poi_present   # 33 tests, hardware-free
# or just `pytest` from repo root for all 168, all three modules
```

**Before trusting any latency number this produces, read "The one
thing to check" below** — it determines whether M3's own accuracy
fixes (and therefore M5's numbers) are actually in effect.

## Run it — no hardware required

```bash
# Fake walkers, built from a hand-written state machine -- fastest
# way to iterate on the viewers:
python3 scripts/run_present.py --config configs/present.example.json \
    --source synthetic --synthetic-mode scripted

# Fake walkers, but driven through the REAL TrackManager and
# measurement code with a physically-realistic D435i error model --
# what covariance shapes, fusion switching, and lifecycle timing
# actually look like:
python3 scripts/run_present.py --config configs/present.example.json \
    --source synthetic --synthetic-mode realistic

# Replay a session already recorded with run_m4.py:
python3 scripts/run_present.py --config configs/present.example.json \
    --source replay --replay-path logs/world_tracks/cam0_....jsonl
```

Then open `http://<this machine's IP>:8765/` — the landing page
links to `/dashboard/` and `/vr/`. On the same machine,
`http://localhost:8765/` works with no token (see Security below).

## Run it live, on the Orin, next to M3+M4

```bash
python3 scripts/run_present.py --config configs/present.example.json \
    --source live \
    --m3-config configs/m3.example.json \
    --m4-config configs/m4.phaseA.example.json \
    --m4-source realsense
```

This builds the *exact* `M4Pipeline` that `run_m4.py` builds and
additionally passes `LiveSource.submit` as its `on_tracks` callback —
a parameter `M4Pipeline` already accepted
(`poi_localization/runtime/m4_pipeline.py`) and `m4_daemon.py` simply
never used. **Zero changes to `poi_localization` or
`poi_perception`.** See `poi_present/sources/live.py`'s docstring and
`scripts/run_present.py`'s `_build_live()` — it's the same
M3-daemon-chains-into-M4-via-`on_detections` pattern `run_m4.py`
already uses, one level further down the chain.

## The one thing to check before trusting any of this

If you've also got a *separate*, standalone `poi_perception` checkout
installed (from before it was folded into this repo), `pip freeze`
may show two installs both providing the top-level `poi_perception`
package:

```bash
python3 -c "import poi_perception; print(poi_perception.__file__)"
```

That should print somewhere under *this* repo. If it prints a
different, standalone repo's path instead, `pip uninstall` that other
one — its copy predates the torso-polygon fix, the
`global_time_enabled` / High Accuracy / post-processing RealSense
tuning, and the extrinsic-uncertainty covariance fix (see this repo's
`CHANGES.md`). Every M5 latency number depends on
`global_time_enabled` actually being on.

## Security

This is a live feed of movement in your home. `configs/present.example.json`
ships with `server.token: null` and `server.host: "0.0.0.0"` for local
development — the server logs a warning on startup if you run it this
way past loopback. Before leaving your desk:

```json
{ "server": { "host": "<this machine's actual LAN IP>", "token": "something-long-and-random" } }
```

Then every URL needs `?token=...` appended
(`http://<ip>:8765/dashboard/?token=...`) — both viewers read it from
the page URL and forward it to the WebSocket automatically.

## What's here

```
poi_present/
  schema.py           # the frozen poi.v1 wire contract (see SCHEMA.md)
  adapter.py           # M4 record -> wire TrackWire, + event derivation
  config.py             # PresentConfig
  sources/
    base.py               # TrackSource interface (one FrameBundle in, out)
    live.py                 # thread-safe bridge from M4Pipeline's on_tracks
    replay.py                # paced JSONL replay of a recorded session
    synthetic.py               # scripted state-machine + "realistic" (real
                                 # TrackManager + sensor model) generators
  server/
    app.py                 # WebSocket + HTTP publisher (single port,
                             # single dependency -- see below)
    clients.py               # per-client bounded queue (drop-oldest tracks,
                               # never-drop events, slow-client disconnect)
    health.py                 # fps / latency / temp / clock-sanity

scripts/run_present.py    # entrypoint: --source live|replay|synthetic
configs/present.example.json

web/
  common/                  # tokens.css, ws-client.js, scene-client.js,
                             # validate.js, vendored three.js + self-hosted fonts
  dashboard/                 # M5a -- 2D top-down radar, track list, events,
                               # health footer, floor-marker overlay
  vr/                          # M5b -- WebXR viewer: dollhouse/life-size,
                                 # trails, uncertainty discs, interpolation,
                                 # Module-1-mesh fallback chain

docs/schema_examples/       # golden poi.v1 messages -- both the Python
                              # tests and the JS validator check against these
SCHEMA.md                   # the wire contract, in prose
tests/test_poi_present/     # 33 tests: schema, adapter/events, health,
                              # client queues, replay/synthetic sources,
                              # and a real end-to-end server test over a socket
```

## Dependency footprint

**One new Python package: `websockets`** (pure Python, zero
dependencies of its own). Its asyncio server answers plain HTTP GETs
via `process_request` as well as WebSocket upgrades, so one process
on one port serves the pages, the vendored JS, the map bundle, and
the socket. `psutil` covers memory; CPU temperature is read directly
from `/sys/class/thermal/thermal_zone*/temp` — deliberately **not**
`nvidia-ml-py`, which doesn't see the Orin's integrated GPU. Both are
already in `requirements.txt`.

**No Node.js and no build step at runtime.** `three.js` (pinned
0.160.0) and IBM Plex Sans/Mono are vendored into
`web/common/vendor/` as plain files, fetched once via `npm` at build
time and committed — the Orin never needs internet, and Three's API
can't drift out from under this build.

## Known gaps / next steps

- **Module 6's calibration watchdog isn't built.** `calib` in the
  `health` message is `"missing"` whenever `frame == "local"` and
  `"ok"` whenever it's `"room"` — there is no `"suspect"` state yet.
- **Camera thumbnail preview** (mentioned as an optional, off-by-
  default feature in the design) isn't implemented on the server
  side; the dashboard has no dead code for it.
- **Dead-zone shading in VR** is wired up as a toggle and a ring
  primitive but isn't computed from the real camera frustum against
  a floor extent yet — it's a placeholder ring, not yet gated on
  Module 1's walkable grid.
- Life-size teleport locomotion is a straight-down raycast from each
  controller to `y=0`; it doesn't yet respect the walkable grid or
  room geometry as a clamp.
- `EventDeriver`'s `id_switch` event (from M4's internal
  `2d_id_mismatch`) isn't emitted — `M4Pipeline` would need a small
  additive `on_events` callback for that, deliberately left out per
  the "M4 stays frozen for now" agreement.
