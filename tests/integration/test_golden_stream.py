"""Frozen end-to-end stream: any edit to M3 or M4 that changes behaviour shows up here.
Intentional change? `python3 tools/make_golden.py`, then say why in docs/CHANGELOG.md."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import make_golden  # noqa: E402

POS_TOL = 5e-3   # metres


@pytest.mark.parametrize("name", sorted(make_golden.SCENARIOS))
def test_stream_matches_golden(name):
    golden = json.loads((ROOT / "tests" / "golden" / f"{name}_phaseA.json").read_text())
    rows = make_golden.run_scenario(name, golden["frames"])
    assert len(rows) == len(golden["rows"]), "frame count changed"
    for got, want in zip(rows, golden["rows"]):
        assert got["frame_id"] == want["frame_id"]
        g = {t["id"]: t for t in got["tracks"]}
        w = {t["id"]: t for t in want["tracks"]}
        assert set(g) == set(w), f"frame {want['frame_id']}: track ids {sorted(g)} != golden {sorted(w)}"
        for i in w:
            assert (g[i]["state"], g[i]["src"]) == (w[i]["state"], w[i]["src"]), f"frame {want['frame_id']} id {i}"
            assert max(abs(a - b) for a, b in zip(g[i]["p"], w[i]["p"])) <= POS_TOL, f"frame {want['frame_id']} id {i} moved"
            # the reported uncertainty is part of the contract: a noise-model edit changes cov_xy long before it moves p
            for a, b in zip(g[i]["cov"], w[i]["cov"]):
                assert abs(a - b) <= 1e-6 + 0.02 * abs(b), f"frame {want['frame_id']} id {i} cov_xy {g[i]['cov']} != {w[i]['cov']}"
            assert (g[i]["h"] is None) == (w[i]["h"] is None)
            if w[i]["h"] is not None:
                assert abs(g[i]["h"] - w[i]["h"]) <= 0.01, f"frame {want['frame_id']} id {i} height"


def test_golden_files_are_not_trivially_empty():
    for name in make_golden.SCENARIOS:
        g = json.loads((ROOT / "tests" / "golden" / f"{name}_phaseA.json").read_text())
        assert sum(1 for r in g["rows"] if r["tracks"]) >= 10, f"{name} golden has almost no tracks"
