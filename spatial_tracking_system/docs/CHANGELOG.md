# Changelog

## 1.0.0 — merge of M1+M2 (`pyslam_live`) and M3+M4+M5 (`spatial_tracking_system`)

**Legacy code changes -- exactly ONE legacy module file differs from what you uploaded** (`python -m sts provenance` proves it;
the other 198 module files, all 6 `run_*.py`/`relocalize.py` entry points and all 10 M3-M5 scripts are byte-identical):
1. `poi_present/server/app.py` -- additive: `calib_provider` attribute, `publish_event_threadsafe()`, `CALIB_STATES` import,
   loop capture in `serve_forever`. With no provider injected, behaviour is exactly as before
   (`test_default_behaviour_is_unchanged_without_a_provider`).

One test file was also edited: `tests/unit/test_poi_present/test_schema_examples.py` -- one path (`parents[2]` -> `parents[3]`)
because the test folder moved one level deeper.

**Layout decisions that avoided editing legacy code.** `run_anchor.py` and `run_live_map.py` locate the default capture profile
relative to their OWN directory (`<script dir>/configs/capture_profile.mapping.json`). They are therefore kept at the repo root
and the profile lives at `configs/capture_profile.mapping.json`, so they resolve it exactly as before. (A first attempt moved the
scripts into an `entrypoints/` folder; that silently fell back to default capture-profile values -- a different hash, which degrades
the map<->anchor `capture_profile_match` gate -- and `pyslam.selftest` failed on `run_bag.py`. Reverted.)
`tests/integration/test_cli_doctor_misc.py::test_legacy_scripts_resolve_the_capture_profile_from_the_new_layout` pins this: if you
move either script, the test fails.

**New:** `sts/` (orchestration), `contracts/`, `tests/integration/`, `tests/synth/site_fixture.py`, `tools/make_golden.py`,
`deploy/` systemd units, `requirements/`, `pyproject.toml` (one package set), `docs/`.

**Findings fixed in the new layer (not in legacy code):** see `docs/SYSTEM_SUMMARY.md` section 9.
