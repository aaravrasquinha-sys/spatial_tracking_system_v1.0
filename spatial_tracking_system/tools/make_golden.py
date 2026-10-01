#!/usr/bin/env python3
"""Regenerate tests/golden/*.json -- the frozen end-to-end stream for the hardware-free scenarios.

The golden stream is what makes "I edited M3/M4 and nothing else changed" checkable: the REAL M3 tracker +
REAL M4 measurement/tracking run over M3's scripted scenarios with a fixed floor frame, and every track's
(id, state, src, position) per frame is compared to this file (tests/integration/test_golden_stream.py).

Regenerate ONLY when a behaviour change is intentional (write why in docs/CHANGELOG.md), and regenerate ON THE
ORIN the first time: aarch64 vs x86 floating point can shift a chi-square gate decision by one frame.

    python3 tools/make_golden.py
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCENARIOS = {"two_crossing": 90, "single_loop": 90, "sitting": 90}


def run_scenario(name: str, frames: int):
    from sts.cli import _synthetic_parts
    from sts.runtime import build_chain, plan_run, run_frames
    from sts.site import SiteConfig
    with tempfile.TemporaryDirectory() as d:
        site = SiteConfig.from_dict({"data_dir": d, "localization": {"phase": "A", "overrides": {
            "phase_a": {"fixed_transform": {"height_m": 2.3, "pitch_deg": 20.0, "yaw_deg": 0.0}},
            "track": {"tentative_confirm_hits": 2}}}})
        plan = plan_run(site, "cam0", write_run_manifest=False)
        src, be = _synthetic_parts(site, "cam0", name, frames)
        chain = build_chain(plan, serve=False, source=src, backend=be, write_logs=False)
        rows = []

        class Rec:
            cam_id, frame_name, map_id = "cam0", "local", None

            def submit(self, frame, records):
                rows.append({"frame_id": frame.frame_id,
                             "tracks": [{"id": r.world_track_id, "state": r.state, "src": r.src,
                                         "p": [round(float(x), 3) for x in r.p_local],
                                         "cov": [float(f"{float(x):.4g}") for x in r.cov_xy],
                                         "h": None if r.height_m is None else round(float(r.height_m), 3)} for r in records]})
        chain.live_source = Rec()
        run_frames(chain, max_frames=frames)
        return rows


def main():
    out = ROOT / "tests" / "golden"
    out.mkdir(exist_ok=True)
    for name, n in SCENARIOS.items():
        rows = run_scenario(name, n)
        (out / f"{name}_phaseA.json").write_text(json.dumps({"scenario": name, "frames": n, "rows": rows}, indent=1))
        print(f"{name}: {len(rows)} frames, {sum(1 for r in rows if r['tracks'])} with tracks")


if __name__ == "__main__":
    main()
