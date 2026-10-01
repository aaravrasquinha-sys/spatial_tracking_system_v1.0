"""
Source-level mutation check for the WP-K (Phase A) and WP-L (Phase B) gates.

    python3 -m pyslam.tools.phase_a_mutation_check          # ~1-2 min

tools/mutation_check.py injects defects into the METRICS' inputs. This one
does something different and blunter: it copies the source tree to a temp
directory, breaks the Phase-A code itself in a specific, realistic way, and
requires tests/gates/test_ga.py to go RED for each break. A gate that stays
green against the bug it exists to catch is worse than no gate, so this is
how the WP-K gates were validated (all four mutants were caught when the
gates were written) and how they should be re-validated after any edit to
run_outputs.py / memory.py / config.py.

Mutants (each is the bug the corresponding fix removed, re-introduced):
  ORDER   map.ply assembled BEFORE pipeline.finalize()          (WP-K1)
  NOCACHE evicted nodes' imagery never written to the cache     (WP-K3)
  RGBBGR  imagery cache written without RGB->BGR conversion     (WP-K3)
  BOOL    "--config-override flag=false" parsed as truthy        (WP-K2)
  GAPOFF  loop_min_path_m / loop_min_time_s silently ignored      (WP-L1)
  LEAK    Bayes diffusion forgets to subtract what it hands on    (WP-L2)
  REDSEL  redundancy policy evicts the LEAST crowded node         (WP-L4)
  ROTFLIP gyro bridge rotation conjugated in the wrong direction  (WP-M)
  NOGROW  translation info stops widening with elapsed gap time   (WP-M)
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _swap_order(text: str) -> str:
    i = text.index("    # 1. closing optimisation")
    j = text.index("    # 2. map, from whatever")
    k = text.index("    # 3. trajectory + plots")
    return text[:i] + text[j:k] + text[i:j] + text[k:]


# (name, file to break, patch function, gate module that must go red)
MUTANTS = [
    ("ORDER", "pyslam/tools/run_outputs.py", _swap_order, "tests.gates.test_ga"),
    ("NOCACHE", "pyslam/memory/memory.py",
     lambda t: t.replace("            self._cache_imagery(node)  # WP-K3: BEFORE the RAM copy is dropped\n", "            pass\n"),
     "tests.gates.test_ga"),
    ("RGBBGR", "pyslam/memory/memory.py",
     lambda t: t.replace("cv2.imwrite(rgb_p, cv2.cvtColor(sig.rgb, cv2.COLOR_RGB2BGR), self._PNG_PARAMS)",
                         "cv2.imwrite(rgb_p, sig.rgb, self._PNG_PARAMS)"), "tests.gates.test_ga"),
    ("BOOL", "pyslam/core/config.py",
     lambda t: t.replace('    if t == "bool":\n        v = raw.strip().lower()', '    if t == "bool":\n        return bool(raw)\n        v = raw.strip().lower()'),
     "tests.gates.test_ga"),
    ("GAPOFF", "pyslam/pipeline.py",
     lambda t: t.replace("        min_path, min_time = self.cfg.loop_min_path_m, self.cfg.loop_min_time_s\n",
                         "        min_path, min_time = 0.0, 0.0\n"), "tests.gates.test_gl"),
    ("LEAK", "pyslam/loop/bayes.py",
     lambda t: t.replace("                    stay -= out\n", "                    pass\n"), "tests.gates.test_gl"),
    ("REDSEL", "pyslam/memory/memory.py",
     lambda t: t.replace("(-crowd[nid], weight_of[nid], nid)", "(crowd[nid], weight_of[nid], nid)"), "tests.gates.test_gl"),
    ("ROTFLIP", "pyslam/imu/bridge.py",
     lambda t: t.replace("return R_body_cam.T @ delta_R_body @ R_body_cam",
                         "return R_body_cam @ delta_R_body @ R_body_cam.T"), "tests.gates.test_gm"),
    ("NOGROW", "pyslam/imu/bridge.py",
     lambda t: t.replace(
         "sigma_p = velocity_sigma_base_mps + velocity_sigma_growth_mps_per_s * max(dt, 0.0)",
         "sigma_p = velocity_sigma_base_mps"), "tests.gates.test_gm"),
]


def main() -> int:
    n_caught = 0
    for name, rel, fn, gate in MUTANTS:
        tmp = tempfile.mkdtemp(prefix=f"pa_mut_{name}_")
        try:
            for d in ("pyslam", "tests"):
                shutil.copytree(os.path.join(ROOT, d), os.path.join(tmp, d),
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            path = os.path.join(tmp, rel)
            with open(path) as f:
                src = f.read()
            mutated = fn(src)
            if mutated == src:
                print(f"  [BROKEN HARNESS] {name}: patch did not apply to {rel} (source drifted?)")
                return 2
            with open(path, "w") as f:
                f.write(mutated)
            r = subprocess.run([sys.executable, "-m", gate], cwd=tmp,
                               capture_output=True, text=True)
            failed = [l for l in r.stdout.splitlines() if "[FAIL]" in l]
            if r.returncode != 0 and failed:
                n_caught += 1
                print(f"  [caught] {name}: {len(failed)} gate(s) went red ({failed[0].split(':')[0].strip()} ...)")
            else:
                print(f"  [MISSED] {name}: gates stayed green against a re-introduced bug")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{n_caught}/{len(MUTANTS)} mutants caught")
    return 0 if n_caught == len(MUTANTS) else 1


if __name__ == "__main__":
    sys.exit(main())
