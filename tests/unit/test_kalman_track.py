import numpy as np

from poi_localization.tracking.kalman_track import KalmanTrack


def test_predict_grows_uncertainty():
    kt = KalmanTrack(0.0, 0.0, process_accel_noise=1.5)
    var0 = kt.P[0, 0]
    kt.predict(dt=0.1)
    assert kt.P[0, 0] > var0


def test_update_shrinks_uncertainty():
    kt = KalmanTrack(0.0, 0.0, process_accel_noise=1.5)
    kt.predict(dt=0.1)
    var_before = kt.P[0, 0]
    kt.update(np.array([0.05, 0.02]), R=np.diag([0.01, 0.01]))
    assert kt.P[0, 0] < var_before


def test_converges_to_true_position_with_repeated_noisy_measurements():
    rng = np.random.default_rng(0)
    true_pos = np.array([3.0, -1.5])
    kt = KalmanTrack(0.0, 0.0, process_accel_noise=0.5)
    R = np.diag([0.02**2, 0.02**2])
    for _ in range(200):
        kt.predict(dt=0.033)
        z = true_pos + rng.normal(scale=0.02, size=2)
        kt.update(z, R)
    assert np.allclose(kt.predicted_position(), true_pos, atol=0.05)


def test_tracks_constant_velocity_motion():
    true_v = np.array([0.5, 0.0])  # m/s
    kt = KalmanTrack(0.0, 0.0, process_accel_noise=1.0)
    R = np.diag([0.01**2, 0.01**2])
    dt = 0.05
    t = 0.0
    pos = np.array([0.0, 0.0])
    for _ in range(300):
        kt.predict(dt)
        pos = pos + true_v * dt
        t += dt
        kt.update(pos, R)
    assert np.allclose(kt.predicted_position(), pos, atol=0.1)
    assert np.allclose(kt.x[2:4], true_v, atol=0.15)


def test_mahalanobis_distance_zero_at_predicted_position():
    kt = KalmanTrack(1.0, 2.0, process_accel_noise=1.0)
    z = np.array(kt.predicted_position())
    d2 = kt.mahalanobis_distance_sq(z, R=np.diag([0.1, 0.1]))
    assert d2 < 1e-9


def test_mahalanobis_distance_large_for_far_measurement():
    kt = KalmanTrack(0.0, 0.0, process_accel_noise=0.5)
    kt.P = np.diag([0.01, 0.01, 1.0, 1.0])  # tight position uncertainty
    z = np.array([5.0, 5.0])
    d2 = kt.mahalanobis_distance_sq(z, R=np.diag([0.01, 0.01]))
    assert d2 > 100  # way outside a 99% chi-square gate (9.21 for 2 dof)


def test_covariance_stays_symmetric_after_many_updates():
    rng = np.random.default_rng(1)
    kt = KalmanTrack(0.0, 0.0, process_accel_noise=1.5)
    for _ in range(500):
        kt.predict(0.033)
        z = rng.normal(size=2)
        kt.update(z, R=np.diag([0.05, 0.05]))
    assert np.allclose(kt.P, kt.P.T, atol=1e-8)
    eigvals = np.linalg.eigvalsh(kt.P)
    assert np.all(eigvals > -1e-8)  # positive semi-definite
