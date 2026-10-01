# Contracts

Modules talk to each other **only** through the contracts in this folder. A contract is a file on disk or a
small in-process interface; everything else inside a module is free to change.

| Contract | Producer | Consumer(s) | Kind | Schema / spec |
|---|---|---|---|---|
| Map bundle | M1 (`pyslam`) | M2, `sts` | files | [`map_bundle.md`](map_bundle.md), `schemas/map_manifest` |
| Anchor bundle | M2 (`pyslam.anchor`) | M4, M5, watchdog, `sts` | files | [`anchor_bundle.md`](anchor_bundle.md), `schemas/calibration`, `walkable`, `room_frame` |
| `Frame`, `Intrinsics` | capture sources | M3, M4, watchdog | in-process | `poi_perception/contracts.py` (mirrors `pyslam.core.types`) |
| `Detection2D` | M3 | M4 | in-process + JSONL | `poi_perception/io/detection2d.py` |
| `FloorFrameTransform` + `FrameProvider` | M4 frames | M4 measurement/tracking | in-process protocol | `poi_localization/frames/` |
| World-track records | M4 | M5, replay, eval | in-process + JSONL | `poi_localization/io/world_track.py` |
| `poi.v1` wire schema | M5 | dashboard, VR viewer, external clients | WebSocket + HTTP | [`runtime_streams.md`](runtime_streams.md), `schemas/poi_*`, `docs/schema_examples/` |
| `camera_model.<cam>.json` | you | M2 + M4 (via `sts`) | file | `schemas/camera_model` |
| `site.json` | you | `sts` | file | `sts/site.py` (every field has a default) |

## Rules

1. **Validated at both ends.** `tests/integration/test_contracts.py` checks what each producer writes AND what each
   consumer loads against the same schema. A change on one side fails CI, not the Orin.
2. **Additive changes only.** Consumers ignore fields they do not recognise (M4's calibration loader already does).
   Removing or renaming a field is a *new contract version*; the consumer declares which versions it accepts
   ([`../docs/COMPATIBILITY.md`](../docs/COMPATIBILITY.md)).
3. **Two types are duplicated on purpose:** `Frame`/`Intrinsics` exist in both `pyslam` and `poi_perception` so M3 never
   imports `pyslam`. `test_frame_and_intrinsics_types_stay_field_compatible` fails if they drift.
4. **The `map_id` rule** ([`map_bundle.md`](map_bundle.md#the-map_id-rule)) is the one contract whose violation silently
   invalidates calibrations. Read it before writing any map finalizer.

`python -m sts contracts --file X.json --schema calibration` validates a file (needs `jsonschema`, a dev dependency).
