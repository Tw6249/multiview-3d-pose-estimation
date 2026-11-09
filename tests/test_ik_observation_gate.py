"""Causal quality-gate behavior and constraint-based G1 arm freezing."""
import json
from pathlib import Path

import numpy as np
import pytest

from g1_imitation.core.constants import KEYPOINT_INDEX_BY_NAME as K
from g1_imitation.core.ik_observation_gate import ArmObservationGate, GatedKeypointIKRetargeter
from g1_imitation.core.s45c_keypoint_ik import KeypointIKRetargeter
from g1_imitation.core.synthetic_human import synthesize_human_27_pose


CONFIG = json.loads((Path(__file__).resolve().parents[1] /
                     "g1_imitation/config/ik_observation_gate_20261109.json").read_text())


def test_gate_calibration_outlier_missing_and_recovery_preserve_input():
    gate = ArmObservationGate(CONFIG)
    pose = synthesize_human_27_pose()
    for timestamp in (0., .1, .2):
        gate.inspect(pose, timestamp, True)
    bad = pose.copy()
    bad[K["right_wrist"]] += [1., 0., 0.]
    snapshot = bad.copy()
    result = gate.inspect(bad, .3, False)
    assert "fore_length_outlier" in result["right"]["reasons"]
    assert not result["left"]["reasons"]
    assert np.array_equal(snapshot, bad)
    bad[K["right_wrist"]] = np.nan
    assert gate.inspect(bad, .4, False)["right"]["reasons"]
    assert not gate.inspect(pose, .5, False)["right"]["reasons"]
    with pytest.raises(ValueError):
        gate.inspect(pose, .5, False)


def test_straight_elbow_keeps_swivel_unknown():
    gate = ArmObservationGate(CONFIG)
    pose = synthesize_human_27_pose(t_pose=True)
    for timestamp in (0., .1, .2):
        gate.inspect(pose, timestamp, True)
    result = gate.inspect(pose, .3, False)
    assert not result["left"]["swivel_observable"]


def test_valid_inputs_match_baseline_and_future_does_not_change_prefix():
    baseline = KeypointIKRetargeter()
    gated = GatedKeypointIKRetargeter(CONFIG, model=baseline.model)
    pose = synthesize_human_27_pose()
    for index in range(50):
        timestamp = index / 30.
        original = baseline.step(pose, timestamp)
        result = gated.step(pose, timestamp)
        np.testing.assert_array_equal(result.q, original.q)
    snapshot = result.q.copy()
    future = pose.copy()
    future[K["left_wrist"]] += [2., 0., 0.]
    gated.step(future, 50 / 30.)
    np.testing.assert_array_equal(result.q, snapshot)


def test_invalid_arm_is_frozen_inside_qp_while_valid_root_moves():
    gated = GatedKeypointIKRetargeter(CONFIG)
    pose = synthesize_human_27_pose()
    for index in range(50):
        previous = gated.step(pose, index / 30.)
    original_ranges = gated.model.jnt_range.copy()
    previous_q = previous.q.copy()
    for index in range(50, 57):
        bad = pose.copy()
        bad[:, 0] += .02 * (index - 49)
        bad[K["right_wrist"]] += [1., 0., 0.]
        result = gated.step(bad, index / 30.)
        np.testing.assert_allclose(result.q[-7:], previous_q[-7:], atol=1e-8)
        assert "right_arm_quality_arm_freeze" in result.held_groups
        assert np.max(np.abs(result.dq[-7:])) < 1e-6
    assert result.q[0] > previous_q[0] + .02
    np.testing.assert_array_equal(original_ranges, gated.model.jnt_range)
    recovered = gated.step(pose, 57 / 30.)
    assert "right_arm_quality_arm_freeze" not in recovered.held_groups
