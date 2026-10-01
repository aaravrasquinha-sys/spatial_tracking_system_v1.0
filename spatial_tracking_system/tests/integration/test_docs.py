"""The documentation tables are part of the contract: slots and open items must match the code."""
import re
from pathlib import Path

from sts.slots import SLOTS

D = Path(__file__).resolve().parents[2] / "docs"


def test_every_slot_and_reserved_name_is_documented():
    txt = (D / "ARCHITECTURE.md").read_text()
    for name, spec in SLOTS.items():
        assert f"`{name}`" in txt, f"slot {name} missing from docs/ARCHITECTURE.md"
        assert f"`{spec.default}`" in txt, f"default {spec.default} of {name} missing"
        for r in spec.reserved:
            assert f"`{r}`" in txt, f"reserved name {r} of {name} missing"


def test_open_items_reference_real_test_and_files():
    txt = (D / "OPEN_ITEMS.md").read_text()
    for needle in ("test_known_legacy_defects", "sts map-lock", "K1", "K2", "K3", "H1", "W5"):
        assert needle in txt
    root = Path(__file__).resolve().parents[2]
    for rel in re.findall(r"`((?:sts|poi_present|web|tools|tests)/[\w/\.\-]+\.(?:py|js))`", txt):
        assert (root / rel).exists(), f"docs/OPEN_ITEMS.md references missing file {rel}"


def test_compatibility_states_the_map_id_rule():
    assert "dense/points.ply" in (D / "COMPATIBILITY.md").read_text()
    assert "dense/points.ply" in (D.parent / "contracts" / "map_bundle.md").read_text()


# Mentioned in the docs only to say they were NOT built (docs/OPEN_ITEMS.md / SYSTEM_SUMMARY.md say so explicitly).
NOT_BUILT = {"tools/bag_to_tum.py"}


def _doc_files():
    return sorted(D.glob("*.md")) + [D.parent / "sts" / "MODULE.md"]


def test_every_documented_sts_command_parses_against_the_real_cli():
    """A runbook whose commands don't exist is worse than none. Every `python -m sts ...` line in the
    docs must be accepted by the real argument parser (placeholders such as `...` and `<x>` are skipped)."""
    import shlex
    from sts.cli import build_parser
    ap = build_parser()
    checked, bad = 0, []
    for f in _doc_files():
        txt = re.sub(r"\\\n\s*", " ", f.read_text())
        for line in txt.splitlines():
            m = re.search(r"python3? -m sts\s+(.*)$", line)
            if not m:
                continue
            cmd = re.split(r"\s+#|\s*\|\s*|`", m.group(1))[0].strip().rstrip("&").strip()
            if not cmd or cmd.startswith("…") or any(c in cmd for c in "<>[]|…"):
                continue
            try:
                argv = shlex.split(cmd)
            except ValueError:
                continue
            checked += 1
            try:
                ap.parse_args(argv)
            except SystemExit:
                bad.append(f"{f.name}: python -m sts {cmd}")
    assert checked >= 30, f"only {checked} commands found -- the extraction is broken"
    assert not bad, "documented commands the CLI rejects:\n" + "\n".join(bad)


def test_repo_paths_mentioned_in_the_docs_exist():
    root = D.parent
    # Only paths under directories that ship in the repo; data/ and logs are created at run time.
    pat = re.compile(r"`((?:sts|pyslam|poi_perception|poi_localization|poi_present|web|tools|tests|scripts|contracts|configs|deploy|requirements|docs)"
                     r"/[\w/\.\-]+\.(?:py|js|md|json|txt|sh|service|timer|toml|html))`")
    missing = []
    for f in _doc_files():
        for rel in set(pat.findall(f.read_text())):
            if "*" in rel or "<" in rel:
                continue
            if (root / rel).exists():
                continue
            if rel in NOT_BUILT:                       # documented as "not built" where it is mentioned
                continue
            example = rel.rsplit(".", 1)[0] + ".example." + rel.rsplit(".", 1)[1]
            if (root / example).exists():              # a file the operator creates by copying the shipped example
                continue
            missing.append(f"{f.name}: {rel}")
    assert not missing, "docs mention files that do not exist:\n" + "\n".join(sorted(missing))
