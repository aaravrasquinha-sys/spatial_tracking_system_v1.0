"""
Run the full synthetic gate suite (no hardware / camera required):

    python -m pyslam.selftest

Every module in this codebase should be considered untrustworthy until
this passes. Run it immediately after `git pull` / unzipping, before
touching the camera.
"""
import sys
import time
import traceback

sys.path.insert(0, ".")

from tests.gates.test_g0 import ALL_TESTS
from tests.gates.test_g_traj import ALL_TESTS as ALL_TRAJ_TESTS
from tests.gates.test_ga import ALL_TESTS as ALL_GA_TESTS
from tests.gates.test_gl import ALL_TESTS as ALL_GL_TESTS
from tests.gates.test_gm import ALL_TESTS as ALL_GM_TESTS
from pyslam.tools.mutation_check import ALL_MUTATION_CHECKS


def main() -> int:
    all_checks = (ALL_TESTS + ALL_TRAJ_TESTS + ALL_GA_TESTS + ALL_GL_TESTS + ALL_GM_TESTS
                  + ALL_MUTATION_CHECKS)
    print(f"pySLAM Phase 0/1 self-test -- {len(ALL_TESTS)} gate checks + "
          f"{len(ALL_TRAJ_TESTS)} trajectory-export gate checks (WP-T0-T3) + "
          f"{len(ALL_GA_TESTS)} Phase-A gate checks (WP-K1-K5) + "
          f"{len(ALL_GL_TESTS)} Phase-B gate checks (WP-L1-L4) + "
          f"{len(ALL_GM_TESTS)} Phase-C gate checks (WP-M, LOST bridge) + "
          f"{len(ALL_MUTATION_CHECKS)} mutation-sensitivity checks (G1A.2)\n")
    n_pass, n_fail = 0, 0
    t_start = time.time()
    for test in all_checks:
        name = test.__name__
        t0 = time.time()
        try:
            test()
            n_pass += 1
        except AssertionError as e:
            n_fail += 1
            print(f"  [FAIL] {name}: {e}")
        except Exception:
            n_fail += 1
            print(f"  [ERROR] {name}:")
            traceback.print_exc()
        finally:
            dt = time.time() - t0
            if dt > 1.0:
                print(f"      ({dt:.1f}s)")

    total = time.time() - t_start
    print(f"\n{n_pass}/{len(all_checks)} passed in {total:.1f}s")
    if n_fail > 0:
        print("SELFTEST FAILED -- do not proceed to hardware until this is green.")
        return 1
    print("SELFTEST PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
