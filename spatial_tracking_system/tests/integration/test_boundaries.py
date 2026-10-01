"""Rule 1: modules only talk through contracts. These tests are what make editing a module safe."""
import textwrap

from sts import boundaries


def test_repo_has_no_forbidden_imports():
    assert boundaries.check() == []


def test_nothing_imports_sts():
    assert boundaries.nothing_imports_sts() == []


def test_checker_actually_detects_violations(tmp_path):
    (tmp_path / "poi_perception").mkdir()
    (tmp_path / "poi_perception" / "bad.py").write_text("from pyslam.core import lie\nimport poi_localization\n")
    (tmp_path / "pyslam").mkdir()
    (tmp_path / "pyslam" / "bad2.py").write_text("import sts.paths\n")
    (tmp_path / "poi_localization").mkdir()
    (tmp_path / "poi_localization" / "ok.py").write_text("from poi_perception.contracts import Frame\n")
    v = boundaries.check(tmp_path)
    got = {(f, imp) for f, _, imp in v}
    assert ("poi_perception/bad.py", "pyslam") in got
    assert ("poi_perception/bad.py", "poi_localization") in got
    assert ("pyslam/bad2.py", "sts") in got
    assert not any(f == "poi_localization/ok.py" for f, _ in got)


def test_matrix_matches_the_documented_rule():
    a = boundaries.ALLOWED
    assert a["pyslam"] == set() and a["poi_perception"] == set()
    assert a["poi_localization"] == {"poi_perception"}
    assert "pyslam" not in a["poi_present"] and "sts" not in a["poi_present"]
    assert a["sts"] >= {"pyslam", "poi_perception", "poi_localization", "poi_present"}
