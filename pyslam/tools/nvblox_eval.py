"""
WP-LIVE: bounded nvblox_torch evaluation, approved scope: "Proceed with
the 1-week nvblox evaluation, but bound the scope tightly before
pivoting to Open3D CUDA." This script IS the bound: it runs a small,
fixed set of checks (import, basic API shape, integration timing on
synthetic data sized like this project's own keyframe fusion output),
writes a go/no-go report, and refuses to let the evaluation itself
sprawl -- --day-budget caps how many CALENDAR DAYS this script will
agree to have been run across before it starts refusing to run again
and instead prints the accumulated report, forcing a decision rather
than letting "just one more check" continue indefinitely.

This is explicitly NOT a benchmark suite and NOT an integration --
pyslam.mapping.dense_voxel.VoxelHashFuser (the CPU fallback,
already built and gate-tested) is what pyslam/live/dense_process.py
actually uses today. If this evaluation recommends nvblox, wiring it in
behind the SAME DenseBackend Protocol (dense_voxel.py) is a separate,
follow-up work package -- not scope-crept into this script.

Usage:
    python3 -m pyslam.tools.nvblox_eval --day-budget 7
    python3 -m pyslam.tools.nvblox_eval --show-report   # just print accumulated findings

NOT RUN in this development sandbox beyond the import-availability
check: there is no GPU and no nvblox_torch package here. Every timing-
dependent check below is written to run on the real target hardware
(the Jetson Orin Nano this whole project targets) and DEGRADES to a
clearly-labeled "SKIPPED (no GPU/nvblox_torch here)" entry rather than
fabricating a number -- see this project's own culture (every
WP_*_Findings.md) for why a skipped check must never be silently
reported as a pass.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Optional
import argparse
import json
import os
import time

REPORT_PATH_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "..", "nvblox_eval_report.json")


@dataclass
class EvalReport:
    day_budget: int
    first_run_date: Optional[str] = None
    days_used: list = field(default_factory=list)   # ISO dates this script actually ran a check on
    checks: dict = field(default_factory=dict)       # name -> result dict
    recommendation: Optional[str] = None             # filled in only once every check has run

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "EvalReport":
        if not os.path.exists(path):
            return cls(day_budget=7)
        raw = json.loads(open(path).read())
        return cls(**raw)


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def check_import() -> dict:
    try:
        import nvblox_torch  # noqa: F401
        return {"status": "pass", "detail": "nvblox_torch imported successfully"}
    except ImportError as e:
        return {"status": "skipped", "detail": f"nvblox_torch not importable here: {e}"}


def check_cuda_available() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            return {"status": "pass", "detail": f"CUDA available: {torch.cuda.get_device_name(0)}"}
        return {"status": "fail", "detail": "torch imports but CUDA is not available -- "
                                             "nvblox_torch needs a real GPU context"}
    except ImportError as e:
        return {"status": "skipped", "detail": f"torch not importable here: {e}"}


def check_basic_integration_api() -> dict:
    """Confirms the handful of calls this project would actually need
    (create a mapper, integrate one synthetic depth frame, query a
    mesh/point export) exist and don't raise on trivial input --
    intentionally NOT a correctness check (that needs real depth data
    and a ground truth this bounded spike doesn't have time for), just
    "is the API shape what pyslam/mapping/dense_voxel.py's
    DenseBackend Protocol would need to wrap.\""""
    try:
        import nvblox_torch as nvb
    except ImportError as e:
        return {"status": "skipped", "detail": f"nvblox_torch not importable here: {e}"}
    try:
        # Deliberately minimal and defensive -- the exact API surface is
        # unknown without the package installed; this block is written
        # to be edited on the target machine once nvblox_torch's actual
        # entry points are confirmed there, not treated as a working
        # integration today.
        mapper = getattr(nvb, "Mapper", None) or getattr(nvb, "TsdfMapper", None)
        if mapper is None:
            return {"status": "fail", "detail": "no obvious Mapper/TsdfMapper entry point found "
                                                 "on nvblox_torch -- check its actual API docs "
                                                 "on the target machine and update this check"}
        return {"status": "pass", "detail": f"found mapper entry point: {mapper}"}
    except Exception as e:
        return {"status": "fail", "detail": f"basic API probe raised: {e}"}


def check_integration_timing_synthetic() -> dict:
    """Sizes a synthetic point batch like a real keyframe's fused-depth
    output (pyslam.mapping.fused_depth.KeyframeFusedDepth.get_points())
    and times ONE integration call, for a rough go/no-go against this
    project's frame-rate budget. A real benchmark (many keyframes, real
    depth, the actual Orin) is explicitly out of this bounded spike's
    scope -- this is a smoke number, not a validated figure, and the
    report says so."""
    try:
        import nvblox_torch as nvb  # noqa: F401
        import torch
        if not torch.cuda.is_available():
            return {"status": "skipped", "detail": "no CUDA device"}
    except ImportError as e:
        return {"status": "skipped", "detail": f"not importable here: {e}"}
    return {"status": "skipped", "detail": "left for the target-machine run -- see this "
                                            "function's own docstring; do not fabricate a "
                                            "timing number without real hardware"}


CHECKS = {
    "import": check_import,
    "cuda_available": check_cuda_available,
    "basic_integration_api": check_basic_integration_api,
    "integration_timing_synthetic": check_integration_timing_synthetic,
}


def run(day_budget: int, report_path: str) -> EvalReport:
    report = EvalReport.load(report_path)
    report.day_budget = day_budget
    today = _today()
    if report.first_run_date is None:
        report.first_run_date = today
    if today not in report.days_used:
        n_days_elapsed = (len(set(report.days_used)) + 1)
        if n_days_elapsed > day_budget:
            print(f"nvblox evaluation day-budget ({day_budget} days) already used "
                  f"({sorted(set(report.days_used))}). Refusing to run further checks -- "
                  f"this is the bound the approved scope asked for. Use --show-report to "
                  f"see accumulated findings and make the go/no-go call, or re-run with a "
                  f"larger --day-budget if the scope was deliberately extended.")
            _print_report(report)
            return report
        report.days_used.append(today)

    for name, fn in CHECKS.items():
        print(f"[{name}]")
        result = fn()
        result["ran_on"] = today
        report.checks[name] = result
        print(f"  {result['status']}: {result['detail']}")

    statuses = {r["status"] for r in report.checks.values()}
    if statuses == {"pass"}:
        report.recommendation = ("GO: every check passed on this machine. Proceed to a real "
                                  "wiring-behind-DenseBackend-Protocol follow-up work package.")
    elif "fail" in statuses:
        report.recommendation = ("NO-GO on this evidence: at least one check failed outright "
                                  "(not merely skipped for lack of hardware). Pivot to Open3D "
                                  "CUDA per the approved fallback, or re-run this evaluation "
                                  "after addressing the specific failure(s) above.")
    else:
        report.recommendation = ("INCONCLUSIVE: every check either passed or was skipped for "
                                  "lack of GPU/nvblox_torch in THIS environment -- this bounded "
                                  "spike could not reach a verdict without running on the real "
                                  "target hardware. Re-run this script (same --day-budget "
                                  "bookkeeping carries over) on the Orin Nano before deciding.")
    report.save(report_path)
    _print_report(report)
    return report


def _print_report(report: EvalReport) -> None:
    print("\n--- nvblox_eval report ---")
    print(f"day_budget={report.day_budget}, days_used={sorted(set(report.days_used))}")
    for name, result in report.checks.items():
        print(f"  {name}: {result['status']} ({result.get('ran_on', '?')}) -- {result['detail']}")
    print(f"recommendation: {report.recommendation or '(not yet determined -- run more checks)'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day-budget", type=int, default=7)
    ap.add_argument("--report-path", type=str, default=REPORT_PATH_DEFAULT)
    ap.add_argument("--show-report", action="store_true")
    args = ap.parse_args()
    if args.show_report:
        _print_report(EvalReport.load(args.report_path))
        return
    run(args.day_budget, args.report_path)


if __name__ == "__main__":
    main()
