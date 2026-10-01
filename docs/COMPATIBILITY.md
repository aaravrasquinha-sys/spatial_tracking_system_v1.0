# Compatibility

| Contract | Version | Produced by | Accepted by | Change policy |
|---|---|---|---|---|
| `poi.v1` wire schema | `poi.v1` | `poi_present` | dashboard, VR viewer (`web/common/validate.js`), `sts` tests | additive fields only; removal/rename => `poi.v2` served alongside |
| Calibration JSON | unversioned, shape frozen by M4's loader | `pyslam.anchor.bundle` | `poi_localization.frames.room_frame`, watchdog, `sts` gate | additive only; M4 ignores unknown keys |
| Walkable grid | `{origin, resolution, grid}` | `pyslam.anchor.walkable` | `poi_localization.tracking.gates.load_walkable_grid`, M5 | additive only |
| Map bundle | `dense/points.ply` + `capture_profile.json` (+ `manifest.json`) | `pyslam` | M2 | **the `map_id` rule** |
| `site.json` | `schema: 1` | you | `sts` | every field defaulted; unknown keys preserved |
| World-track JSONL | `phase` field `A`/`B` per row | `poi_localization.io` | `poi_localization.eval`, `poi_present.sources.replay` | Phase A uses `p_local/v_local`, Phase B `p/v` **on purpose** so a local-frame log can never be mistaken for room-frame |
| `sts` slots | see `sts/slots.py` | `sts.adapters` | `sts.runtime`, `sts.cli` | reserved names are refused, not ignored |

## The `map_id` rule (repeated because violating it is silent)

`map_id` = first 12 hex of SHA-256 over `dense/points.ply` and `capture_profile.json`. A finalizer, exporter or cleaner that
edits either file, or hashes a different file set, produces a different id and invalidates every calibration made against
the old one. Different content files = a different map = re-anchor. See `contracts/map_bundle.md`.

## Environment compatibility (your Orin venv, from `pip freeze`)

| Item | Pinned / expected | Why |
|---|---|---|
| Python | 3.10 | JetPack 6 |
| numpy | 1.26.4 | `opencv-python-headless 4.8.1` wheels and the source-built `gtsam` link against numpy 1.x; the legacy docs' "numpy >= 2" does not match this venv and nothing in the code needs it |
| OpenCV | `opencv-python-headless` 4.8.x | no GUI. For live `imshow` swap to `opencv-python` (debug scripts only) |
| websockets | `>=14,<18` | M5 uses the new asyncio API; **not installed in the freeze you sent** |
| pytest-asyncio | dev only | used by the M5 tests; never declared in the legacy requirements |
| pyrealsense2 | source-built RSUSB + CUDA | `setup_orin.sh`; the freeze lists a `2.58.4` package, check it is not shadowing that build (`sts doctor`) |
| tensorrt / pycuda | JetPack's 10.3 | runtime inference path imports only these two |
| torch / ultralytics | export only | `torch 2.14` with `cu13` wheels is not the JetPack build; the runtime path never imports it |
