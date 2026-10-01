"""
WP-P5.0: an IMU-consistent trajectory generator, for oracle-testing
preintegration ONLY -- not (yet) wired into SyntheticSource/the
pipeline. See Phase 5 planning notes: SyntheticSource's existing
`_imu_between` sets `a_specific = R.T @ (-g_world)` -- gravity reaction
ONLY, with no acceleration from actual motion, ever. It cannot validate
preintegration, bias convergence, or anything else Phase 5 needs; this
module exists specifically to fix that, for the oracle first.

Design: position p(t) is a closed-form sum of sinusoids, so velocity
p'(t) and acceleration p''(t) are exact by direct term-by-term
differentiation -- no numerical differencing anywhere in this half.
Angular velocity w_body(t) is likewise specified directly as a smooth
closed-form function (this IS what a gyro measures -- there is no need
to go through an orientation function's derivative at all). Orientation
R(t) is then obtained by integrating R' = R @ skew(w_body(t)) at a
MUCH finer step than the IMU's own sample rate (RK4, 100x finer by
default) -- an accepted numerical ground truth for this kind of ODE
(RK4 local error O(h^5)/global O(h^4); at a 100x-finer step than a
typical 200Hz IMU this is many orders of magnitude below the Euler
preintegration error the oracle is actually trying to measure).
Gyro/accel readings are then read off these two EXACT (resp.
converged-to-double-precision) functions at the desired IMU sample
times -- no finite-differencing of already-discretised poses anywhere,
which is the trap `handheld_poses`' own OU-jitter would set for any
naive "diff the rendered trajectory" approach.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from pyslam.core import lie


@dataclass
class ImuTrajectory:
    """A fully analytic (position) / numerically-converged (rotation)
    ground-truth trajectory, with exact gyro/accel readings available
    at any query time via `sample_imu`."""
    pos_freqs: np.ndarray    # (3,3) angular frequencies per axis/term
    pos_amps: np.ndarray     # (3,3) amplitudes per axis/term
    pos_phases: np.ndarray   # (3,3) phases per axis/term
    p0: np.ndarray           # (3,) DC offset
    gyro_freqs: np.ndarray   # (3,) one term per axis, kept simple
    gyro_amps: np.ndarray    # (3,)
    gyro_phases: np.ndarray  # (3,)
    R0: np.ndarray           # (3,3) orientation at t=0
    g_world: np.ndarray      # (3,) gravity, world frame

    # ---- position (exact, closed-form) ----
    def pos(self, t: float) -> np.ndarray:
        s = np.sin(self.pos_freqs * t + self.pos_phases)
        return self.p0 + np.sum(self.pos_amps * s, axis=1)

    def vel(self, t: float) -> np.ndarray:
        c = np.cos(self.pos_freqs * t + self.pos_phases)
        return np.sum(self.pos_amps * self.pos_freqs * c, axis=1)

    def accel(self, t: float) -> np.ndarray:
        s = np.sin(self.pos_freqs * t + self.pos_phases)
        return -np.sum(self.pos_amps * (self.pos_freqs ** 2) * s, axis=1)

    # ---- angular velocity (exact, closed-form; body frame) ----
    def omega_body(self, t: float) -> np.ndarray:
        return self.gyro_amps * np.sin(self.gyro_freqs * t + self.gyro_phases)

    # ---- orientation (RK4-integrated ground truth) ----
    _FINE_DT = 2e-4  # fixed "ground truth" integration step (5000 Hz),
        # ~25x finer than a typical 200Hz IMU sample gap and >>the
        # discretisation error Euler preintegration is what the oracle
        # is actually trying to measure. Fixed (not scaled per query)
        # so a single continuous pass can serve every checkpoint in one
        # sweep -- see integrate_checkpoints().

    def integrate_checkpoints(self, checkpoints: list[float]) -> dict:
        """ONE forward RK4 integration of R' = R @ skew(w_body(t)) from
        R(0)=R0 to max(checkpoints), recording R at every requested
        checkpoint time as the sweep passes it. O(max(checkpoints) /
        _FINE_DT) total work, done ONCE -- the shared engine behind
        both `rotation(t)` (single checkpoint) and `sample_imu` (many
        checkpoints, one per IMU sample), so neither pays for a fresh
        from-zero integration per query."""
        if not checkpoints:
            return {}
        order = sorted(range(len(checkpoints)), key=lambda i: checkpoints[i])
        sorted_cp = [checkpoints[i] for i in order]
        t_end = sorted_cp[-1]
        h = self._FINE_DT
        n = max(1, int(np.ceil(t_end / h))) if t_end > 0 else 0
        h = t_end / n if n > 0 else h

        R = self.R0.copy()
        tt = 0.0
        out = {}
        ci = 0
        if ci < len(sorted_cp) and sorted_cp[ci] <= 0.0:
            out[sorted_cp[ci]] = R.copy()
            ci += 1
        for _ in range(n):
            w1 = self.omega_body(tt)
            k1 = R @ lie.skew(w1)
            w2 = self.omega_body(tt + h / 2)
            k2 = (R + 0.5 * h * k1) @ lie.skew(w2)
            k3 = (R + 0.5 * h * k2) @ lie.skew(w2)
            w4 = self.omega_body(tt + h)
            k4 = (R + h * k3) @ lie.skew(w4)
            R = R + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            tt += h
            while ci < len(sorted_cp) and sorted_cp[ci] <= tt + 1e-12:
                U, _, Vt = np.linalg.svd(R)
                Rn = U @ Vt
                if np.linalg.det(Rn) < 0:
                    U[:, -1] *= -1
                    Rn = U @ Vt
                out[sorted_cp[ci]] = Rn
                ci += 1
        return out

    def rotation(self, t: float) -> np.ndarray:
        return self.integrate_checkpoints([t])[t]

    def pose(self, t: float) -> np.ndarray:
        return lie.make_T(self.rotation(t), self.pos(t))

    def sample_imu(self, t0: float, t1: float, hz: float) -> np.ndarray:
        """Exact (noise-free) gyro/accel samples at `hz`, spanning
        (t0, t1] -- same convention as Frame.imu/SyntheticSource's own
        'since previous frame' semantics. Returns (N,7): t, gx,gy,gz,
        ax,ay,az (accel = specific force in body frame, i.e. what a
        real accelerometer reads: R(t)^T @ (accel_world(t) - g_world))."""
        dt = 1.0 / hz
        n = max(1, int(round((t1 - t0) * hz)))
        times = [t0 + i * dt for i in range(1, n + 1)]
        R_at = self.integrate_checkpoints(times)  # ONE pass, not one per sample
        rows = []
        for t in times:
            w = self.omega_body(t)
            R = R_at[t]
            a_specific = R.T @ (self.accel(t) - self.g_world)
            rows.append([t, *w, *a_specific])
        return np.array(rows, dtype=np.float64)


def make_imu_trajectory(seed: int = 0) -> ImuTrajectory:
    """A single fixed, deliberately non-trivial (multi-axis, multi-
    frequency, not axis-aligned) trajectory for oracle use -- not
    randomised per-call the way scenario builders are, since the oracle
    wants a FIXED, reproducible, once-inspected trajectory, not a
    distribution of them."""
    rng = np.random.default_rng(seed)
    pos_freqs = np.array([[0.9, 1.7, 0.31], [0.6, 2.3, 0.47], [1.1, 0.4, 3.1]])
    pos_amps = np.array([[0.8, 0.15, 0.05], [0.6, 0.2, 0.04], [0.3, 0.1, 0.03]])
    pos_phases = rng.uniform(0, 2 * np.pi, size=(3, 3))
    p0 = np.array([0.0, 0.0, 1.0])

    gyro_freqs = np.array([1.3, 0.9, 1.6])
    gyro_amps = np.radians(np.array([25.0, 18.0, 12.0]))  # rad/s peak
    gyro_phases = rng.uniform(0, 2 * np.pi, size=3)

    R0 = np.eye(3)
    g_world = np.array([0.0, 0.0, -9.81])
    return ImuTrajectory(pos_freqs, pos_amps, pos_phases, p0,
                          gyro_freqs, gyro_amps, gyro_phases, R0, g_world)
