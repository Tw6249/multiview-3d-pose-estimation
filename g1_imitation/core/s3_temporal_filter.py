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

    def reset(self) -> None:
        self.x_hat = None
        self.x_prev_raw = None
        self.dx_hat = None

    def filter_frame(self, points_3d_raw: np.ndarray, dt: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Filter one frame of shape (N, 3), handling NaN entries via zero-order hold + velocity decay."""
        x = np.asarray(points_3d_raw, dtype=np.float64)
        step_dt = self.dt if dt is None else float(dt)

        if self.x_hat is None or self.dx_hat is None or self.x_prev_raw is None:
            self.x_hat = np.where(np.isfinite(x), x, 0.0).copy()
            self.x_prev_raw = self.x_hat.copy()
            self.dx_hat = np.zeros_like(self.x_hat)
            return self.x_hat.copy(), self.dx_hat.copy()

        valid_joint = np.all(np.isfinite(x), axis=1)
        alpha_d = float(smoothing_alpha(self.f_c_d, step_dt))

        # Raw velocity for valid joints: dX_t = (X_t - X_{t-1}) / dt (Eq. 1)
        dx_raw = np.zeros_like(self.x_hat)
        dx_raw[valid_joint] = (x[valid_joint] - self.x_prev_raw[valid_joint]) / step_dt

        # Update smoothed derivative dx_hat (Eq. 1)
        new_dx_hat = self.dx_hat.copy()
        new_dx_hat[valid_joint] = alpha_d * dx_raw[valid_joint] + (1.0 - alpha_d) * self.dx_hat[valid_joint]
        new_dx_hat[~valid_joint] = self.nan_velocity_decay * self.dx_hat[~valid_joint]

        # Adaptive cutoff frequency f_{c,t} = f_{c_min} + beta * |dx_hat_t| (Eq. 2)
        f_c = self.f_c_min + self.beta * np.abs(new_dx_hat)
        alpha = smoothing_alpha(f_c, step_dt)

        # Filtered position update (Eq. 3)
        new_x_hat = self.x_hat.copy()
        new_x_hat[valid_joint] = (
            alpha[valid_joint] * x[valid_joint] + (1.0 - alpha[valid_joint]) * self.x_hat[valid_joint]
        )
        # Invalid (NaN) joints hold previous x_hat_{t-1}

        self.x_prev_raw[valid_joint] = x[valid_joint]
        self.x_hat = new_x_hat
        self.dx_hat = new_dx_hat
        return self.x_hat.copy(), self.dx_hat.copy()


class TemporalFilter:
    """Module S3 temporal filter: Vectorized One-Euro Filter + ground non-penetration clamp (`Z >= z_nominal`)."""

    def __init__(
        self,
        freq_hz: float = 100.0,
        f_c_min: float = 1.2,
        beta: float = 0.04,
        f_c_d: float = 1.0,
        z_nominal_m: float = 0.0,
    ) -> None:
        self.freq_hz = float(freq_hz)
        self.dt = 1.0 / self.freq_hz
        self.z_nominal_m = float(z_nominal_m)
        self.one_euro = VectorizedOneEuroFilter3D(
            freq_hz=freq_hz,
            f_c_min=f_c_min,
            beta=beta,
            f_c_d=f_c_d,
        )

    def reset(self) -> None:
        self.one_euro.reset()

    def process_frame(self, points_3d_raw: np.ndarray, dt: float | None = None) -> TemporalFilterResult:
        step_dt = self.dt if dt is None else float(dt)
        raw = np.asarray(points_3d_raw, dtype=np.float64)
        if raw.shape != (27, 3):
            raise ValueError(f"Expected (27, 3) input to Module S3, got {raw.shape}")

        pts_out, vels_out = self.one_euro.filter_frame(raw, dt=step_dt)
        pts_out[:, 2] = np.maximum(pts_out[:, 2], self.z_nominal_m)

        return TemporalFilterResult(
            points_3d_smooth_m=pts_out,
            velocities_ms=vels_out,
        )


# Backward-compatible alias for imports across scripts/tests
TemporalContactFilter = TemporalFilter
