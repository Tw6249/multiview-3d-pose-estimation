"""Module S3: Adaptive Vectorized 3D One-Euro Temporal Filter.

Specification Reference: `docs/方案规范.md` — 模块 S3 (with discrete Schmitt foot-lock/soft-unlock removed)
Input:
  - `points_3d_raw`: np.ndarray of shape (27, 3), raw triangulated 3D keypoints in world Z-up frame (metres).
Output:
  - `TemporalFilterResult`:
    - `points_3d_smooth_m`: np.ndarray of shape (27, 3), smoothed & floor-clamped 3D keypoints (metres).
    - `velocities_ms`: np.ndarray of shape (27, 3), smoothed 3D keypoint velocities (m/s).
Failure Behavior:
  - If any joint is `NaN` at frame t, holds previous filtered position `X_hat_{t-1}` and decays velocity
    `dX_hat_t = 0.8 * dX_hat_{t-1}` (Section S3.2.1).
  - Enforces hard floor non-penetration constraint `Z >= z_nominal` so no keypoint penetrates `Z < 0`.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


def smoothing_alpha(cutoff_hz: np.ndarray | float, dt: float) -> np.ndarray | float:
    """Compute first-order low-pass filter coefficient alpha(f_c, dt) = 1 / (1 + 1 / (2*pi*f_c*dt))."""
    tau = 1.0 / (2.0 * math.pi * np.asarray(cutoff_hz, dtype=np.float64))
    return 1.0 / (1.0 + tau / float(dt))


@dataclass(frozen=True)
class TemporalFilterResult:
    """Output of Module S3 for a single frame."""
    points_3d_smooth_m: np.ndarray
    velocities_ms: np.ndarray
    fresh_mask: np.ndarray | None = None
    age_s: np.ndarray | None = None


class VectorizedOneEuroFilter3D:
    """Vectorized 3D One-Euro Filter for (27, 3) skeleton trajectories (Section S3.2.1)."""

    def __init__(
        self,
        freq_hz: float = 100.0,
        f_c_min: float = 1.2,
        beta: float = 0.04,
        f_c_d: float = 1.0,
        nan_velocity_decay: float = 0.8,
    ) -> None:
        self.freq_hz = float(freq_hz)
        self.dt = 1.0 / self.freq_hz
        self.f_c_min = float(f_c_min)
        self.beta = float(beta)
        self.f_c_d = float(f_c_d)
        self.nan_velocity_decay = float(nan_velocity_decay)

        self.x_hat: np.ndarray | None = None
        self.x_prev_raw: np.ndarray | None = None
        self.dx_hat: np.ndarray | None = None
        self.initialized: np.ndarray | None = None

    def reset(self) -> None:
        self.x_hat = None
        self.x_prev_raw = None
        self.dx_hat = None
        self.initialized = None

    def filter_frame(self, points_3d_raw: np.ndarray, dt: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Filter one frame of shape (N, 3), handling NaN entries via zero-order hold + velocity decay."""
        x = np.asarray(points_3d_raw, dtype=np.float64)
        step_dt = self.dt if dt is None else float(dt)
        if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(step_dt) or step_dt <= 0.0:
            raise ValueError("points must be (N,3) and dt must be finite and positive")

        if self.x_hat is None or self.dx_hat is None or self.x_prev_raw is None:
            valid = np.all(np.isfinite(x), axis=1)
            self.x_hat = np.full_like(x, np.nan)
            self.x_hat[valid] = x[valid]
            self.x_prev_raw = self.x_hat.copy()
            self.dx_hat = np.zeros_like(self.x_hat)
            self.initialized = valid.copy()
            return self.x_hat.copy(), self.dx_hat.copy()

        valid_joint = np.all(np.isfinite(x), axis=1)
        new_joint = valid_joint & ~self.initialized
        regular_joint = valid_joint & self.initialized
        alpha_d = float(smoothing_alpha(self.f_c_d, step_dt))

        # Raw velocity for valid joints: dX_t = (X_t - X_{t-1}) / dt (Eq. 1)
        dx_raw = np.zeros_like(self.x_hat)
        dx_raw[regular_joint] = (x[regular_joint] - self.x_prev_raw[regular_joint]) / step_dt

        # Update smoothed derivative dx_hat (Eq. 1)
        new_dx_hat = self.dx_hat.copy()
        new_dx_hat[regular_joint] = (alpha_d * dx_raw[regular_joint]
                                     + (1.0 - alpha_d) * self.dx_hat[regular_joint])
        new_dx_hat[new_joint] = 0.0
        new_dx_hat[~valid_joint] = self.nan_velocity_decay * self.dx_hat[~valid_joint]

        # Adaptive cutoff frequency f_{c,t} = f_{c_min} + beta * |dx_hat_t| (Eq. 2)
        f_c = self.f_c_min + self.beta * np.abs(new_dx_hat)
        alpha = smoothing_alpha(f_c, step_dt)

        # Filtered position update (Eq. 3)
        new_x_hat = self.x_hat.copy()
        new_x_hat[regular_joint] = (
            alpha[regular_joint] * x[regular_joint]
            + (1.0 - alpha[regular_joint]) * self.x_hat[regular_joint]
        )
        new_x_hat[new_joint] = x[new_joint]
        # Invalid (NaN) joints hold previous x_hat_{t-1}

        self.x_prev_raw[valid_joint] = x[valid_joint]
        self.initialized[valid_joint] = True
        self.x_hat = new_x_hat
        self.dx_hat = new_dx_hat
        return self.x_hat.copy(), self.dx_hat.copy()


class TemporalFilter:
    """Module S3 main temporal filter: 3x3-covariance Anisotropic Kalman Filter (`kalman_3x3`, default),
    Spherical Skeleton Kalman Filter (`spherical_kalman`), or legacy One-Euro Filter (`one_euro`),
    or legacy One-Euro Filter, plus ground non-penetration clamp (`Z >= z_nominal`).
    """

    def __init__(
        self,
        freq_hz: float = 100.0,
        f_c_min: float = 1.2,
        beta: float = 0.04,
        f_c_d: float = 1.0,
        z_nominal_m: float = 0.0,
        reset_gap_s: float = 0.15,
        *,
        mode: str = "kalman_3x3",
        max_hold_s: float = 0.08,
        default_obs_sigma_m: float = 0.015,
    ) -> None:
        if mode not in ("spherical_kalman", "kalman_3x3", "one_euro"):
            raise ValueError(f"Unsupported TemporalFilter mode {mode!r}")
        self.freq_hz = float(freq_hz)
        self.dt = 1.0 / self.freq_hz
        self.z_nominal_m = float(z_nominal_m)
        self.reset_gap_s = float(reset_gap_s)
        if not np.isfinite(self.reset_gap_s) or self.reset_gap_s <= 0.0:
            raise ValueError("reset_gap_s must be finite and positive")
        self.mode = str(mode)
        self.one_euro = VectorizedOneEuroFilter3D(
            freq_hz=freq_hz,
            f_c_min=f_c_min,
            beta=beta,
            f_c_d=f_c_d,
        )
        self.kalman_3x3 = AnisotropicKalmanFilter3D(
            max_hold_s=max_hold_s,
            reset_gap_s=reset_gap_s,
            default_obs_sigma_m=default_obs_sigma_m,
        )
        self.spherical_kalman = SphericalSkeletonKalmanFilter3D(
            max_hold_s=max_hold_s,
            reset_gap_s=reset_gap_s,
            default_obs_sigma_m=default_obs_sigma_m,
        )
        self._last_timestamp_s: float | None = None
        self._synthetic_timestamp_s: float = 0.0

    def reset(self) -> None:
        self.one_euro.reset()
        self.kalman_3x3.reset()
        self.spherical_kalman.reset()
        self._last_timestamp_s = None
        self._synthetic_timestamp_s = 0.0

    def process_frame(
        self,
        points_3d_raw: np.ndarray,
        dt: float | None = None,
        timestamp_s: float | None = None,
        covariance_m2: np.ndarray | None = None,
    ) -> TemporalFilterResult:
        raw = np.asarray(points_3d_raw, dtype=np.float64)
        if raw.shape != (27, 3):
            raise ValueError(f"Expected (27, 3) input to Module S3, got {raw.shape}")
        next_timestamp = None
        if timestamp_s is not None:
            timestamp_s = float(timestamp_s)
            if not np.isfinite(timestamp_s):
                raise ValueError("timestamp_s must be finite")
            if self._last_timestamp_s is None:
                step_dt = self.dt if dt is None else float(dt)
            else:
                measured_dt = timestamp_s - self._last_timestamp_s
                if measured_dt <= 0.0:
                    raise ValueError("timestamps must be strictly increasing")
                if dt is not None and not np.isclose(float(dt), measured_dt, rtol=1e-5, atol=1e-9):
                    raise ValueError("dt conflicts with timestamp_s interval")
                step_dt = measured_dt
                if measured_dt > self.reset_gap_s:
                    self.one_euro.reset()
            next_timestamp = timestamp_s
        else:
            step_dt = self.dt if dt is None else float(dt)
            if not np.isfinite(step_dt) or step_dt <= 0.0:
                raise ValueError("dt must be finite and positive")
            if self._last_timestamp_s is None:
                next_timestamp = self._synthetic_timestamp_s
            else:
                next_timestamp = self._last_timestamp_s + step_dt

        fresh_mask = None
        age_s = None
        if self.mode == "one_euro":
            pts_out, vels_out = self.one_euro.filter_frame(raw, dt=step_dt)
            fresh_mask = np.all(np.isfinite(raw), axis=1)
            age_s = np.where(fresh_mask, 0.0, step_dt)
        elif self.mode == "kalman_3x3":
            res = self.kalman_3x3.process_frame(raw, next_timestamp, covariance_m2)
            pts_out, vels_out, fresh_mask, age_s = (
                res.points_3d_m.copy(),
                res.velocities_ms.copy(),
                res.fresh_mask,
                res.age_s,
            )
        else:
            res = self.spherical_kalman.process_frame(raw, next_timestamp, covariance_m2)
            pts_out, vels_out, fresh_mask, age_s = (
                res.points_3d_m.copy(),
                res.velocities_ms.copy(),
                res.fresh_mask,
                res.age_s,
            )

        finite_z = np.isfinite(pts_out[:, 2])
        pts_out[finite_z, 2] = np.maximum(pts_out[finite_z, 2], self.z_nominal_m)
        if next_timestamp is not None:
            self._last_timestamp_s = next_timestamp

        return TemporalFilterResult(
            points_3d_smooth_m=pts_out,
            velocities_ms=vels_out,
            fresh_mask=fresh_mask,
            age_s=age_s,
        )


# Backward-compatible alias for imports across scripts/tests
TemporalContactFilter = TemporalFilter


@dataclass(frozen=True)
class LowLatencyFilterResult:
    points_3d_m: np.ndarray
    velocities_ms: np.ndarray
    fresh_mask: np.ndarray
    age_s: np.ndarray


class CausalAlphaBetaFilter3D:
    """Variable-time alpha-beta filter for 3D points with explicit stale-data output.

    Missing measurements hold the previous estimate for `max_hold_s`, then return
    NaN. The filter never extrapolates an unobserved pose into a new output.
    """

    def __init__(self, *, alpha: float = 0.82, beta: float = 0.12, max_hold_s: float = 0.08,
                 reset_gap_s: float = 0.15, uncertainty_scale_m: float = 0.03) -> None:
        if not 0.0 < alpha <= 1.0 or not 0.0 <= beta <= 1.0:
            raise ValueError("alpha must be in (0,1], beta in [0,1]")
        if max_hold_s < 0.0 or reset_gap_s <= 0.0 or uncertainty_scale_m <= 0.0:
            raise ValueError("hold, reset and uncertainty durations must be positive")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_hold_s = float(max_hold_s)
        self.reset_gap_s = float(reset_gap_s)
        self.uncertainty_scale_m = float(uncertainty_scale_m)
        self.reset()

    def reset(self) -> None:
        self._x = np.full((27, 3), np.nan, dtype=np.float64)
        self._v = np.zeros((27, 3), dtype=np.float64)
        self._last_measurement_s = np.full(27, np.nan, dtype=np.float64)
        self._last_timestamp_s: float | None = None

    def process_frame(self, points_3d_raw: np.ndarray, timestamp_s: float,
                      covariance_m2: np.ndarray | None = None) -> LowLatencyFilterResult:
        x = np.asarray(points_3d_raw, dtype=np.float64)
        t = float(timestamp_s)
        if x.shape != (27, 3) or not np.isfinite(t):
            raise ValueError("expected finite timestamp and points with shape (27,3)")
        if self._last_timestamp_s is not None and t <= self._last_timestamp_s:
            raise ValueError("timestamps must be strictly increasing")
        cov = None if covariance_m2 is None else np.asarray(covariance_m2, dtype=np.float64)
        if cov is not None and cov.shape != (27, 3, 3):
            raise ValueError("covariance_m2 must have shape (27,3,3)")

        fresh = np.all(np.isfinite(x), axis=1)
        for j in np.flatnonzero(fresh):
            if not np.isfinite(self._last_measurement_s[j]):
                self._x[j] = x[j]
                self._v[j] = 0.0
            else:
                dt = t - self._last_measurement_s[j]
                if dt > self.reset_gap_s:
                    self._x[j] = x[j]
                    self._v[j] = 0.0
                else:
                    predicted = self._x[j] + self._v[j] * dt
                    residual = x[j] - predicted
                    quality = 1.0
                    if cov is not None and np.isfinite(cov[j]).all():
                        trace = max(0.0, float(np.trace(cov[j])))
                        scale2 = self.uncertainty_scale_m ** 2
                        quality = scale2 / (scale2 + trace)
                    a = self.alpha * quality
                    b = self.beta * quality
                    self._x[j] = predicted + a * residual
                    self._v[j] += (b / dt) * residual
            self._last_measurement_s[j] = t

        age = np.full(27, np.inf, dtype=np.float64)
        seen = np.isfinite(self._last_measurement_s)
        age[seen] = t - self._last_measurement_s[seen]
        available = seen & (age <= self.max_hold_s)
        out = np.full((27, 3), np.nan, dtype=np.float64)
        vel = np.full((27, 3), np.nan, dtype=np.float64)
        out[available] = self._x[available]
        vel[available] = self._v[available]
        self._last_timestamp_s = t
        return LowLatencyFilterResult(out, vel, fresh, age)


def _regularize_cov3x3(
    cov: np.ndarray | None,
    default_sigma_m: float = 0.015,
    min_sigma_m: float = 0.002,
    max_sigma_m: float = 0.25,
) -> np.ndarray:
    """Symmetrize and clamp eigenvalues of a 3x3 covariance matrix (m^2) for numerical stability."""
    if cov is None:
        return (default_sigma_m ** 2) * np.eye(3, dtype=np.float64)
    c = np.asarray(cov, dtype=np.float64)
    if c.shape != (3, 3) or not np.isfinite(c).all():
        return (default_sigma_m ** 2) * np.eye(3, dtype=np.float64)
    sym = 0.5 * (c + c.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    clamped = np.clip(eigvals, min_sigma_m ** 2, max_sigma_m ** 2)
    return (eigvecs * clamped) @ eigvecs.T


class AnisotropicKalmanFilter3D:
    """Causal 6-state (position + damped velocity) Kalman filter using full 3x3 S2 covariance matrices.

    Each joint maintains a 6D state [x, y, z, vx, vy, vz] and a full 6x6 covariance matrix P_t.
    Measurement updates use the anisotropic 3x3 triangulation covariance R_t = Sigma_t directly,
    so high-variance camera-ray directions are smoothed more heavily than well-constrained
    lateral directions. During rapid acceleration, innovation-based maneuver adaptation inflates
    process covariance to prevent phase lag without inventing motion during missing frames.
    """

    def __init__(
        self,
        *,
        accel_noise_m_s2: float = 4.5,
        pos_process_noise_m_s: float = 0.008,
        velocity_decay_tau_s: float = 0.06,
        maneuver_chi2_threshold: float = 3.5,
        maneuver_gain: float = 12.0,
        max_hold_s: float = 0.08,
        reset_gap_s: float = 0.15,
        default_obs_sigma_m: float = 0.015,
    ) -> None:
        if accel_noise_m_s2 <= 0.0 or pos_process_noise_m_s < 0.0 or velocity_decay_tau_s <= 0.0:
            raise ValueError("process noise and velocity decay parameters must be positive")
        if max_hold_s < 0.0 or reset_gap_s <= 0.0 or default_obs_sigma_m <= 0.0:
            raise ValueError("hold, reset and observation sigma parameters must be positive")
        self.accel_noise_m_s2 = float(accel_noise_m_s2)
        self.pos_process_noise_m_s = float(pos_process_noise_m_s)
        self.velocity_decay_tau_s = float(velocity_decay_tau_s)
        self.maneuver_chi2_threshold = float(maneuver_chi2_threshold)
        self.maneuver_gain = float(maneuver_gain)
        self.max_hold_s = float(max_hold_s)
        self.reset_gap_s = float(reset_gap_s)
        self.default_obs_sigma_m = float(default_obs_sigma_m)
        self.reset()

    def reset(self) -> None:
        self._x = np.full((27, 3), np.nan, dtype=np.float64)
        self._v = np.zeros((27, 3), dtype=np.float64)
        self._P = np.zeros((27, 6, 6), dtype=np.float64)
        self._last_measurement_s = np.full(27, np.nan, dtype=np.float64)
        self._last_timestamp_s: float | None = None
        self._last_regularized_R = np.broadcast_to(
            (self.default_obs_sigma_m ** 2) * np.eye(3, dtype=np.float64), (27, 3, 3)
        ).copy()

    def _init_joint(self, j: int, pos: np.ndarray, R: np.ndarray) -> None:
        self._x[j] = pos
        self._v[j] = 0.0
        self._P[j].fill(0.0)
        self._P[j, :3, :3] = R
        self._P[j, 3:, 3:] = (1.5 ** 2) * np.eye(3, dtype=np.float64)

    def process_frame(
        self,
        points_3d_raw: np.ndarray,
        timestamp_s: float,
        covariance_m2: np.ndarray | None = None,
    ) -> LowLatencyFilterResult:
        x = np.asarray(points_3d_raw, dtype=np.float64)
        t = float(timestamp_s)
        if x.shape != (27, 3) or not np.isfinite(t):
            raise ValueError("expected finite timestamp and points with shape (27,3)")
        if self._last_timestamp_s is not None and t <= self._last_timestamp_s:
            raise ValueError("timestamps must be strictly increasing")
        cov = None if covariance_m2 is None else np.asarray(covariance_m2, dtype=np.float64)
        if cov is not None and cov.shape != (27, 3, 3):
            raise ValueError("covariance_m2 must have shape (27,3,3)")

        I3 = np.eye(3, dtype=np.float64)
        I6 = np.eye(6, dtype=np.float64)
        fresh = np.all(np.isfinite(x), axis=1)
        fresh_idx = np.flatnonzero(fresh)

        if len(fresh_idx):
            R_batch = np.empty((len(fresh_idx), 3, 3), dtype=np.float64)
            if cov is None:
                R_batch[:] = (self.default_obs_sigma_m ** 2) * I3
            else:
                cov_sub = cov[fresh_idx]
                finite_cov = np.isfinite(cov_sub).all(axis=(1, 2))
                R_batch[~finite_cov] = (self.default_obs_sigma_m ** 2) * I3
                if np.any(finite_cov):
                    sym = 0.5 * (cov_sub[finite_cov] + np.swapaxes(cov_sub[finite_cov], -1, -2))
                    w, v = np.linalg.eigh(sym)
                    w_clamped = np.clip(w, 0.002 ** 2, 0.25 ** 2)
                    R_batch[finite_cov] = (v * w_clamped[:, None, :]) @ np.swapaxes(v, -1, -2)
            self._last_regularized_R[fresh_idx] = R_batch

            last_t = self._last_measurement_s[fresh_idx]
            is_init = ~np.isfinite(last_t) | ((t - last_t) > self.reset_gap_s)
            for local_i in np.flatnonzero(is_init):
                self._init_joint(int(fresh_idx[local_i]), x[fresh_idx[local_i]], R_batch[local_i])

            upd_local = np.flatnonzero(~is_init)
            if len(upd_local):
                upd_j = fresh_idx[upd_local]
                R_u = R_batch[upd_local]
                dts = t - self._last_measurement_s[upd_j]
                gammas = np.exp(-dts / self.velocity_decay_tau_s)
                eff_dts = self.velocity_decay_tau_s * (1.0 - gammas)

                x_pred = self._x[upd_j] + eff_dts[:, None] * self._v[upd_j]
                v_pred = gammas[:, None] * self._v[upd_j]

                P_prev = self._P[upd_j]
                P00 = P_prev[:, :3, :3]
                P01 = P_prev[:, :3, 3:]
                P11 = P_prev[:, 3:, 3:]
                ed = eff_dts[:, None, None]
                gm = gammas[:, None, None]
                sa2 = self.accel_noise_m_s2 ** 2
                sp2 = self.pos_process_noise_m_s ** 2
                dt1 = dts[:, None, None]
                q_pp = (sa2 * (dt1 ** 3) / 3.0 + sp2 * dt1) * I3
                q_pv = (sa2 * (dt1 ** 2) / 2.0) * I3
                q_vv = (sa2 * dt1) * I3

                P_pred = np.empty_like(P_prev)
                P_pred[:, :3, :3] = P00 + ed * (P01 + np.swapaxes(P01, -1, -2)) + (ed * ed) * P11 + q_pp
                P_pred[:, :3, 3:] = gm * (P01 + ed * P11) + q_pv
                P_pred[:, 3:, :3] = np.swapaxes(P_pred[:, :3, 3:], -1, -2)
                P_pred[:, 3:, 3:] = (gm * gm) * P11 + q_vv

                innovation = x[upd_j] - x_pred
                S0 = P_pred[:, :3, :3] + R_u
                _, eigvecs = np.linalg.eigh(R_u)
                proj_innov = np.einsum("nji,nj->ni", eigvecs, innovation)
                sk = np.einsum("nji,njk,nki->ni", eigvecs, S0, eigvecs)
                nis_k = (proj_innov * proj_innov) / np.maximum(sk, 1e-12)
                gains = np.where(
                    nis_k > self.maneuver_chi2_threshold,
                    self.maneuver_gain * (nis_k / self.maneuver_chi2_threshold - 1.0),
                    0.0,
                )
                inflated_mask = np.any(gains > 0.0, axis=1)
                if np.any(inflated_mask):
                    dir_scale = np.einsum("nik,nk,njk->nij", eigvecs[inflated_mask], gains[inflated_mask], eigvecs[inflated_mask])
                    dt_inf = dt1[inflated_mask]
                    P_pred[inflated_mask, :3, :3] += (sa2 * (dt_inf ** 3) / 3.0 + sp2 * dt_inf) * dir_scale
                    q_pv_dir = (sa2 * (dt_inf ** 2) / 2.0) * dir_scale
                    P_pred[inflated_mask, :3, 3:] += q_pv_dir
                    P_pred[inflated_mask, 3:, :3] += q_pv_dir
                    P_pred[inflated_mask, 3:, 3:] += (sa2 * dt_inf) * dir_scale
                    S0[inflated_mask] = P_pred[inflated_mask, :3, :3] + R_u[inflated_mask]

                K = np.swapaxes(np.linalg.solve(np.swapaxes(S0, -1, -2), np.swapaxes(P_pred[:, :, :3], -1, -2)), -1, -2)
                corr = np.einsum("nij,nj->ni", K, innovation)
                self._x[upd_j] = x_pred + corr[:, :3]
                self._v[upd_j] = v_pred + corr[:, 3:]

                KH = np.zeros((len(upd_j), 6, 6), dtype=np.float64)
                KH[:, :, :3] = K
                I_KH = I6 - KH
                P_upd = I_KH @ P_pred @ np.swapaxes(I_KH, -1, -2) + K @ R_u @ np.swapaxes(K, -1, -2)
                self._P[upd_j] = 0.5 * (P_upd + np.swapaxes(P_upd, -1, -2))

            self._last_measurement_s[fresh_idx] = t

        age = np.full(27, np.inf, dtype=np.float64)
        seen = np.isfinite(self._last_measurement_s)
        age[seen] = t - self._last_measurement_s[seen]
        available = seen & (age <= self.max_hold_s)
        out = np.full((27, 3), np.nan, dtype=np.float64)
        vel = np.full((27, 3), np.nan, dtype=np.float64)
        out[available] = self._x[available]
        vel[available] = self._v[available]
        self._last_timestamp_s = t
        return LowLatencyFilterResult(out, vel, fresh, age)


class _SphericalBoneTrack:
    """State for a single bone segment: unit direction u in S^2, tangent velocity u_dot, and length L."""

    def __init__(self) -> None:
        self.u: np.ndarray | None = None
        self.u_dot = np.zeros(3, dtype=np.float64)
        self.P_u = np.zeros((6, 6), dtype=np.float64)
        self.length: float = np.nan
        self.length_var: float = np.nan
        self.last_t: float = np.nan


class SphericalSkeletonKalmanFilter3D:
    """Skeleton Kalman filter combining 3x3 anisotropic root filtering with S^2 bone-direction filtering.

    - Root (`mid_hip`) and non-chain extremities are tracked with `AnisotropicKalmanFilter3D`.
    - Core skeleton segments (pelvis axis, spine, shoulder axis, upper arms, forearms, thighs, shanks)
      decompose the relative observation vector `d = p_b - p_a` and its endpoint covariance
      `Sigma_d = Sigma_a + Sigma_b` into:
        1. Radial bone-length observation `r = ||d||` with variance `u^T Sigma_d u`, filtered by a 1D
           Kalman filter so ray-depth stretching is suppressed without freezing bone length blindly.
        2. Unit direction `u = d / r` on S^2 with tangent-projected 3x3 covariance
           `R_u = J_u Sigma_d J_u^T`, predicted along great circles via Rodrigues rotation of the
           angular velocity `omega = u x u_dot` so rotary limb swing does not overshoot tangentially.
    """

    _BONES: tuple[tuple[str, int, int], ...] = (
        ("left_upper", 5, 7),
        ("right_upper", 6, 8),
        ("left_fore", 7, 9),
        ("right_fore", 8, 10),
        ("left_thigh", 11, 13),
        ("right_thigh", 12, 14),
        ("left_shank", 13, 15),
        ("right_shank", 14, 16),
    )

    def __init__(
        self,
        *,
        angular_accel_noise_rad_s2: float = 18.0,
        direction_process_noise_rad_s: float = 0.025,
        angular_velocity_decay_tau_s: float = 0.055,
        length_process_noise_m_s: float = 0.004,
        maneuver_chi2_threshold: float = 3.5,
        maneuver_gain: float = 12.0,
        max_hold_s: float = 0.08,
        reset_gap_s: float = 0.15,
        default_obs_sigma_m: float = 0.015,
    ) -> None:
        if angular_accel_noise_rad_s2 <= 0.0 or angular_velocity_decay_tau_s <= 0.0 or length_process_noise_m_s <= 0.0:
            raise ValueError("spherical filter process parameters must be positive")
        self.angular_accel_noise_rad_s2 = float(angular_accel_noise_rad_s2)
        self.direction_process_noise_rad_s = float(direction_process_noise_rad_s)
        self.angular_velocity_decay_tau_s = float(angular_velocity_decay_tau_s)
        self.length_process_noise_m_s = float(length_process_noise_m_s)
        self.maneuver_chi2_threshold = float(maneuver_chi2_threshold)
        self.maneuver_gain = float(maneuver_gain)
        self.max_hold_s = float(max_hold_s)
        self.reset_gap_s = float(reset_gap_s)
        self.default_obs_sigma_m = float(default_obs_sigma_m)
        self._point_kf = AnisotropicKalmanFilter3D(
            max_hold_s=max_hold_s,
            reset_gap_s=reset_gap_s,
            default_obs_sigma_m=default_obs_sigma_m,
        )
        self.reset()

    def reset(self) -> None:
        self._point_kf.reset()
        self._bones: dict[str, _SphericalBoneTrack] = {
            name: _SphericalBoneTrack()
            for name in ("hip_axis", "spine", "shoulder_axis", *(b[0] for b in self._BONES))
        }
        self._prev_out: np.ndarray | None = None
        self._prev_t: float | None = None

    def _update_bone(self, name: str, vec_obs: np.ndarray, cov_d: np.ndarray, t: float) -> np.ndarray | None:
        track = self._bones[name]
        if not np.isfinite(vec_obs).all():
            if track.u is not None and np.isfinite(track.last_t) and (t - track.last_t) <= self.max_hold_s:
                return track.length * track.u
            return None
        r_obs = float(np.linalg.norm(vec_obs))
        if not np.isfinite(r_obs) or r_obs < 1e-4:
            return None
        u_obs = vec_obs / r_obs
        I3 = np.eye(3, dtype=np.float64)
        I6 = np.eye(6, dtype=np.float64)
        proj_perp = I3 - np.outer(u_obs, u_obs)

        # Radial variance along the bone axis and 3x3 directional covariance on T_u S^2
        var_r = float(np.clip(u_obs @ cov_d @ u_obs, 1e-6, 0.05 ** 2))
        J_u = proj_perp / max(r_obs, 0.05)
        R_u = J_u @ cov_d @ J_u.T + (0.008 ** 2) * proj_perp + (0.02 ** 2) * np.outer(u_obs, u_obs)
        R_u = 0.5 * (R_u + R_u.T)
        eigvals, eigvecs = np.linalg.eigh(R_u)
        eigvals = np.clip(eigvals, 0.005 ** 2, 0.50 ** 2)
        R_u = (eigvecs * eigvals) @ eigvecs.T

        if track.u is None or not np.isfinite(track.last_t) or (t - track.last_t) > self.reset_gap_s:
            track.u = u_obs.copy()
            track.u_dot = np.zeros(3, dtype=np.float64)
            track.P_u.fill(0.0)
            track.P_u[:3, :3] = R_u
            track.P_u[3:, 3:] = (4.0 ** 2) * I3
            track.length = r_obs
            track.length_var = var_r
            track.last_t = t
            return track.length * track.u

        dt = t - track.last_t
        # 1D Kalman update on bone length
        l_pred_var = track.length_var + (self.length_process_noise_m_s ** 2) * dt
        k_l = l_pred_var / (l_pred_var + var_r)
        track.length = float(track.length + k_l * (r_obs - track.length))
        track.length_var = float((1.0 - k_l) * l_pred_var)

        # Great-circle prediction on S^2 using angular velocity omega = u x u_dot
        gamma = math.exp(-dt / self.angular_velocity_decay_tau_s)
        eff_dt = self.angular_velocity_decay_tau_s * (1.0 - gamma)
        omega = np.cross(track.u, track.u_dot)
        theta = float(np.linalg.norm(omega) * eff_dt)
        if theta > 1e-7:
            axis = omega / np.linalg.norm(omega)
            # Rodrigues rotation of track.u around `axis` by `theta`
            u_pred = (
                math.cos(theta) * track.u
                + math.sin(theta) * np.cross(axis, track.u)
                + (1.0 - math.cos(theta)) * float(np.dot(axis, track.u)) * axis
            )
            u_pred /= max(float(np.linalg.norm(u_pred)), 1e-12)
        else:
            u_pred = track.u.copy()
        omega_pred = gamma * omega
        u_dot_pred = np.cross(omega_pred, u_pred)

        F = np.block([[I3, eff_dt * I3], [np.zeros((3, 3)), gamma * I3]])
        sa2 = self.angular_accel_noise_rad_s2 ** 2
        su2 = self.direction_process_noise_rad_s ** 2
        q_uu = (sa2 * (dt ** 3) / 3.0 + su2 * dt) * I3
        q_uw = (sa2 * (dt ** 2) / 2.0) * I3
        q_ww = (sa2 * dt) * I3
        Q = np.block([[q_uu, q_uw], [q_uw, q_ww]])
        P_pred = F @ track.P_u @ F.T + Q

        innovation = u_obs - u_pred
        S0 = P_pred[:3, :3] + R_u
        dir_scale = np.zeros((3, 3), dtype=np.float64)
        inflated = False
        for axis_idx in range(3):
            ek = eigvecs[:, axis_idx]
            sk = float(ek @ S0 @ ek)
            proj_innov = float(ek @ innovation)
            nis_k = (proj_innov * proj_innov) / max(sk, 1e-12)
            if nis_k > self.maneuver_chi2_threshold:
                inflated = True
                gain_k = self.maneuver_gain * (nis_k / self.maneuver_chi2_threshold - 1.0)
                dir_scale += gain_k * np.outer(ek, ek)
        if inflated:
            Q_dir = np.block([
                [(sa2 * (dt ** 3) / 3.0 + su2 * dt) * dir_scale, (sa2 * (dt ** 2) / 2.0) * dir_scale],
                [(sa2 * (dt ** 2) / 2.0) * dir_scale, (sa2 * dt) * dir_scale],
            ])
            P_pred = P_pred + Q_dir
            S0 = P_pred[:3, :3] + R_u

        K = np.linalg.solve(S0.T, P_pred[:, :3].T).T
        state_upd = np.r_[u_pred, u_dot_pred] + K @ innovation
        u_new = state_upd[:3]
        u_norm = float(np.linalg.norm(u_new))
        u_new = u_obs.copy() if u_norm < 1e-8 else (u_new / u_norm)
        # Project tangential velocity onto T_{u_new} S^2
        u_dot_new = state_upd[3:] - float(np.dot(state_upd[3:], u_new)) * u_new

        KH = np.zeros((6, 6), dtype=np.float64)
        KH[:, :3] = K
        P_upd = (I6 - KH) @ P_pred @ (I6 - KH).T + K @ R_u @ K.T

        track.u = u_new
        track.u_dot = u_dot_new
        track.P_u = 0.5 * (P_upd + P_upd.T)
        track.last_t = t
        return track.length * track.u

    def process_frame(
        self,
        points_3d_raw: np.ndarray,
        timestamp_s: float,
        covariance_m2: np.ndarray | None = None,
    ) -> LowLatencyFilterResult:
        base_res = self._point_kf.process_frame(points_3d_raw, timestamp_s, covariance_m2)
        x = np.asarray(points_3d_raw, dtype=np.float64)
        t = float(timestamp_s)
        R_reg = self._point_kf._last_regularized_R
        fresh = base_res.fresh_mask

        out = base_res.points_3d_m.copy()
        # Reconstruct pelvis, spine, shoulders, and 8 limb segments via spherical bone tracks when anchors exist
        hips_ok = bool(fresh[11] and fresh[12])
        if hips_ok:
            mid_hip = 0.5 * (out[11] + out[12])
            raw_hip_vec = x[11] - x[12]
            hip_vec = self._update_bone("hip_axis", raw_hip_vec, R_reg[11] + R_reg[12], t)
            if hip_vec is not None:
                out[11] = mid_hip + 0.5 * hip_vec
                out[12] = mid_hip - 0.5 * hip_vec

            sh_raw_ok = bool(fresh[5] and fresh[6])
            raw_spine = 0.5 * (x[5] + x[6]) - 0.5 * (x[11] + x[12]) if sh_raw_ok else np.full(3, np.nan)
            cov_spine = 0.25 * (R_reg[5] + R_reg[6] + R_reg[11] + R_reg[12])
            spine_vec = self._update_bone("spine", raw_spine, cov_spine, t)
            raw_sh_vec = x[5] - x[6] if sh_raw_ok else np.full(3, np.nan)
            sh_vec = self._update_bone("shoulder_axis", raw_sh_vec, R_reg[5] + R_reg[6], t)
            if sh_raw_ok and spine_vec is not None and sh_vec is not None:
                mid_sh = mid_hip + spine_vec
                out[5] = mid_sh + 0.5 * sh_vec
                out[6] = mid_sh - 0.5 * sh_vec

        for bone_name, idx_a, idx_b in self._BONES:
            raw_vec = (
                x[idx_b] - x[idx_a]
                if (fresh[idx_a] and fresh[idx_b])
                else np.full(3, np.nan)
            )
            bone_vec = self._update_bone(bone_name, raw_vec, R_reg[idx_a] + R_reg[idx_b], t)
            if bone_vec is not None and fresh[idx_a] and fresh[idx_b] and np.isfinite(out[idx_a]).all():
                out[idx_b] = out[idx_a] + bone_vec

        vel = base_res.velocities_ms.copy()
        if self._prev_out is not None and self._prev_t is not None and (t - self._prev_t) <= self.reset_gap_s:
            dt = t - self._prev_t
            both_ok = np.isfinite(out).all(axis=1) & np.isfinite(self._prev_out).all(axis=1)
            vel[both_ok] = (out[both_ok] - self._prev_out[both_ok]) / dt
        self._prev_out = out.copy()
        self._prev_t = t
        return LowLatencyFilterResult(out, vel, base_res.fresh_mask, base_res.age_s)
