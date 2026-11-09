"""Control-rate driver: causality, freshness bookkeeping and q/dq consistency."""
import json
from pathlib import Path

import numpy as np

from g1_imitation.core.control_rate_reference import retarget_at_control_rate
from g1_imitation.core.s45c_keypoint_ik import KeypointIKRetargeter
from g1_imitation.core.s6_mujoco_closed_loop import generate_10_diverse_human_motions
from g1_imitation.core.structured_history_ik import StructuredHistoryRetargeter

CONFIG = json.loads((Path(__file__).resolve().parents[1] /
                     "g1_imitation/config/structured_history_20261109.json").read_text())


def _walk():
    walk = generate_10_diverse_human_motions(duration_s=3.0, fps_hz=100.0)["01_forward_walking_gait"][::3]
    return np.arange(len(walk)) / 30.0 + 0.004, walk


def test_ticks_use_only_past_observations_and_report_age():
    times, walk = _walk()
    out = retarget_at_control_rate(KeypointIKRetargeter(), times, walk, 0.02)
    assert np.all(times[out["source_index"]] <= out["t"] + 1e-12)
    assert np.all(out["observation_age_s"] >= 0) and out["observation_age_s"].max() < 1 / 30 + 1e-9
    assert out["fresh"].sum() == len(np.unique(out["source_index"]))


def test_prefix_is_independent_of_future_observations():
    times, walk = _walk()
    retargeter = KeypointIKRetargeter()
    full = retarget_at_control_rate(retargeter, times, walk, 0.02)["q"]
    part = retarget_at_control_rate(retargeter, times[:60], walk[:60], 0.02)["q"]
    np.testing.assert_array_equal(full[:len(part)], part)


def test_history_bounds_hold_on_delivered_ticks_and_dq_matches_q():
    times, walk = _walk()
    retargeter = StructuredHistoryRetargeter(CONFIG, structured=False, history=True)
    out = retarget_at_control_rate(retargeter, times, walk, 0.02)
    live = out["observed"][1:] & out["observed"][:-1]
    velocity = np.diff(out["q"][:, 7:], axis=0) / 0.02
    np.testing.assert_allclose(velocity[live], out["dq"][1:, 6:][live], atol=1e-9)
    after = out["t"][1:] >= retargeter._t_calib_end + retargeter.cfg["startup_blend_s"] + 0.02
    assert np.abs(velocity[live & after]).max() <= CONFIG["history"]["velocity_limit_rad_s"] + 1e-6


def test_prediction_uses_filter_motion_model_and_zero_velocity_equals_hold():
    times, walk = _walk()
    hold = retarget_at_control_rate(KeypointIKRetargeter(), times, walk, 0.02)
    still = retarget_at_control_rate(KeypointIKRetargeter(), times, walk, 0.02,
                                     velocities=np.zeros_like(walk), velocity_decay_tau_s=0.06)
    np.testing.assert_array_equal(hold["q"], still["q"])
    captured = []

    class Recorder(KeypointIKRetargeter):
        def step(self, points, t_s):
            captured.append(np.asarray(points).copy())
            return super().step(points, t_s)

    velocity = np.zeros_like(walk)
    velocity[:, :, 0] = 1.0
    out = retarget_at_control_rate(Recorder(), times, walk, 0.02, velocities=velocity, velocity_decay_tau_s=0.06)
    expected = 0.06 * (1 - np.exp(-out["observation_age_s"] / 0.06))
    shifts = np.array([c[5, 0] for c in captured]) - walk[out["source_index"], 5, 0]
    np.testing.assert_allclose(shifts, expected, atol=1e-12)
