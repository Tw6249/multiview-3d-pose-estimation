"""Causal control-rate reference generation: one IK solve per control tick.

Input : retargeter with `step(points (27,3), t_s)` returning q/dq/observed; observation capture times (T,) s,
        strictly increasing; observed points (T,27,3) m, right-handed Z-up world; control period s.
Output: dict with tick times (N,) s, q (N, nq), dq (N, nv) = backward difference of the emitted q at the tick
        period, observed (N,), fresh (N,) True when a new observation arrived since the previous tick,
        observation_age_s (N,) = tick time minus capture time of the observation used, solve_ms (N,).
Method: a tick uses the newest observation captured at or before it (zero-order hold of the OBSERVATION, never
        of the robot reference) and re-solves IK from the previous tick's solution. Any rate constraint inside
        the retargeter therefore holds at the delivered control period. No future observation is read.
Prediction (optional): with the S3 filter's own velocities (T,27,3) m/s and its velocity decay time constant
        tau s, each point is moved by tau * (1 - exp(-age / tau)) * v, the same motion model the filter uses
        between measurements. Points without a finite velocity are held. Prediction is a model output, not a
        measurement: `fresh` is unchanged and `predicted_horizon_s` reports the horizon.
Failure: ValueError for shape/time errors; retargeter errors propagate. Held observations are not new data:
        `fresh` reports them; without prediction this module never extrapolates a pose.
"""
from __future__ import annotations

import numpy as np


def retarget_at_control_rate(retargeter, capture_times, points, control_dt, velocities=None,
                             velocity_decay_tau_s=None, max_prediction_s=0.08):
    capture_times = np.asarray(capture_times, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if capture_times.ndim != 1 or points.shape != (len(capture_times), 27, 3) or len(capture_times) < 2:
        raise ValueError("expected capture times (T,) and points (T,27,3) with T >= 2")
    if not np.all(np.diff(capture_times) > 0) or not np.isfinite(capture_times).all():
        raise ValueError("capture times must be finite and strictly increasing")
    if not np.isfinite(control_dt) or control_dt <= 0:
        raise ValueError("control period must be positive")
    if velocities is not None:
        velocities = np.asarray(velocities, dtype=np.float64)
        if velocities.shape != points.shape or not velocity_decay_tau_s or velocity_decay_tau_s <= 0:
            raise ValueError("velocities must match points and the decay time constant must be positive")
    ticks = np.arange(capture_times[0], capture_times[-1] + 1e-9, float(control_dt))
    source = np.searchsorted(capture_times, ticks + 1e-9, side="right") - 1
    age = ticks - capture_times[source]
    if velocities is not None and age.max() > max_prediction_s:
        raise ValueError("observation gap exceeds the allowed prediction horizon")
    retargeter.reset()
    frames = []
    for tick, index, horizon in zip(ticks, source, age):
        frame_points = points[index]
        if velocities is not None and horizon > 0:
            gain = velocity_decay_tau_s * (1.0 - np.exp(-horizon / velocity_decay_tau_s))
            shift = gain * velocities[index]
            frame_points = frame_points + np.where(np.isfinite(shift), shift, 0.0)
        frames.append(retargeter.step(frame_points, float(tick)))
    return dict(
        t=ticks,
        source_index=source,
        q=np.array([frame.q for frame in frames]),
        dq=np.array([frame.dq for frame in frames]),
        observed=np.array([frame.observed for frame in frames]),
        fresh=np.r_[True, source[1:] != source[:-1]],
        observation_age_s=age,
        predicted_horizon_s=age if velocities is not None else np.zeros_like(age),
        solve_ms=np.array([frame.solve_ms for frame in frames]),
        held=[frame.held_groups for frame in frames],
    )
