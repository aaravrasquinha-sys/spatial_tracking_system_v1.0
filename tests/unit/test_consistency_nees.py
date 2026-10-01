"""
Turns poi_localization/eval/consistency.py's NEES check into an
automatic regression guard: if a future change re-introduces the
"covariance ignores extrinsic uncertainty" bug (or any other
overconfidence regression), this test fails in CI without needing
hardware or a human watching a bird's-eye-view window.
"""
from poi_localization.eval.consistency import run_consistency_check


def test_perfect_calibration_nees_is_not_overconfident():
    result = run_consistency_check(extrinsic_pitch_err_deg=0.0, sigma_rot_deg=0.0, sigma_trans_m=0.0)
    assert result.n_samples > 50
    # Sensor-noise-only case: covariance should be in a sane, non-degenerate
    # band. A median far above this would mean the sensor noise model
    # itself (depth/raycast sigma) is understated -- the same class of bug
    # the extrinsic fix addressed, just for the base noise terms instead.
    assert result.nees_median < 5.0


def test_unmodeled_extrinsic_error_is_still_flagged_as_overconfident():
    """Documents the known limitation, not a bug: if the system has a
    real calibration error but ISN'T told about it (sigma_rot_deg=0
    while extrinsic_pitch_err_deg>0), NEES should be clearly bad. This
    is the exact bug this session's fixes targeted -- pinning it here
    means a future change can't silently regress the "extrinsic
    uncertainty must be threaded through" contract back to the old
    (unmodeled) behavior without this test catching it."""
    result = run_consistency_check(extrinsic_pitch_err_deg=1.0, sigma_rot_deg=0.0, sigma_trans_m=0.0)
    assert result.nees_median > 10.0


def test_honestly_reported_extrinsic_uncertainty_is_much_better_than_unreported():
    """The core claim behind the fix: telling the system what its own
    calibration uncertainty is (even approximately) substantially closes
    the NEES gap versus not telling it at all, for the same real
    calibration error."""
    unreported = run_consistency_check(extrinsic_pitch_err_deg=1.0, sigma_rot_deg=0.0, sigma_trans_m=0.0)
    reported = run_consistency_check(extrinsic_pitch_err_deg=1.0, sigma_rot_deg=1.0, sigma_trans_m=0.02)
    assert reported.nees_median < 0.5 * unreported.nees_median


def test_zero_extrinsic_error_position_accuracy_is_within_acceptance_ballpark():
    """Loose accuracy sanity check (Section 11's floor-marker test is the
    real acceptance gate; this just guards against a gross regression in
    the measurement math under ideal conditions)."""
    result = run_consistency_check(extrinsic_pitch_err_deg=0.0)
    assert result.position_error_median_m < 0.05
