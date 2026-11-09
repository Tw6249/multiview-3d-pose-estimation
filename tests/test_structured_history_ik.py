"""Geometry Jacobians and causal shared-frame constrained IK behavior."""
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from g1_imitation.core.constants import KEYPOINT_INDEX_BY_NAME as K
from g1_imitation.core.s45c_keypoint_ik import KeypointIKRetargeter
from g1_imitation.core.structured_history_ik import (
    BendCosineTask, FrameJointBox, SegmentDirectionTask, StructuredHistoryRetargeter,
)
from g1_imitation.core.synthetic_human import synthesize_human_27_pose

CONFIG = json.loads((Path(__file__).resolve().parents[1] /
                     "g1_imitation/config/structured_history_20261109.json").read_text())


@pytest.mark.parametrize("bend", [False, True])
def test_normalized_geometry_jacobian_matches_manifold_finite_difference(bend):
    retargeter = KeypointIKRetargeter()
    configuration = retargeter.configuration
    chain = CONFIG["geometry"]["chains"]["left_arm"]
    task = BendCosineTask(retargeter.model, chain, 1.) if bend else SegmentDirectionTask(retargeter.model, *chain[:2], 1.)
    original = configuration.q.copy()
    original[7:] = retargeter.default_q + .03 * np.sin(np.arange(retargeter.nj))
    configuration.update(original)
    jacobian = task.compute_jacobian(configuration)
    finite_difference = np.zeros_like(jacobian)
    for column in range(retargeter.nv):
        errors = []
        for sign in (1., -1.):
            tangent = np.zeros(retargeter.nv)
            tangent[column] = sign * 1e-6
            pose = original.copy()
            mujoco.mj_integratePos(retargeter.model, pose, tangent, 1.)
            configuration.update(pose)
            errors.append(task.compute_error(configuration))
        finite_difference[:, column] = (errors[0] - errors[1]) / 2e-6
    np.testing.assert_allclose(jacobian, finite_difference, atol=2e-6)


def test_joint_box_remaining_budget_depends_on_current_iteration_not_new_budget():
    retargeter = KeypointIKRetargeter()
    limit = FrameJointBox(retargeter.model)
    anchor = retargeter.configuration.q.copy()
    limit.set_bounds(anchor[7:], np.zeros(retargeter.nj), .03, 8., 60., retargeter.lo, retargeter.hi)
    bounds = limit.upper.copy()
    constraints = limit.compute_qp_inequalities(retargeter.configuration, 1.)
    pose = anchor.copy()
    pose[7:] += .02
    retargeter.configuration.update(pose)
    next_constraints = limit.compute_qp_inequalities(retargeter.configuration, 1.)
    np.testing.assert_allclose(next_constraints.h[:29], constraints.h[:29] - .02, atol=1e-12)
    np.testing.assert_array_equal(limit.upper, bounds)


def test_disabled_features_match_baseline():
    baseline = KeypointIKRetargeter()
    experiment = StructuredHistoryRetargeter(CONFIG, structured=False, history=False, model=baseline.model)
    points = synthesize_human_27_pose()
    for index in range(60):
        expected = baseline.step(points, index / 30.)
        actual = experiment.step(points, index / 30.)
        np.testing.assert_array_equal(actual.q, expected.q)


def test_history_constraints_prefix_and_model_unchanged_under_abrupt_targets():
    retargeter = StructuredHistoryRetargeter(CONFIG, history=True)
    prefix = StructuredHistoryRetargeter(CONFIG, history=True, model=retargeter.model)
    ranges = retargeter.model.jnt_range.copy()
    previous_velocity = np.zeros(29)
    diagnostics = []
    for index in range(80):
        points = synthesize_human_27_pose(t_pose=index >= 50,
                                         squat_depth_m=.2 if index >= 65 else 0.)
        if 56 <= index < 60:
            points[K["right_wrist"]] = np.nan
        result = retargeter.step(points, index / 30.)
        diagnostics.append(retargeter.experiment_diagnostics)
        if index < 70:
            expected = prefix.step(points, index / 30.)
            np.testing.assert_allclose(result.q, expected.q, atol=1e-12)
        if diagnostics[-1]["history_active"]:
            assert np.max(np.abs(result.dq[6:])) <= 8. + 1e-6
            assert np.max(np.abs(result.dq[6:] - previous_velocity)) / (1 / 30.) <= 60. + 1e-5
            previous_velocity = result.dq[6:].copy()
        else:
            previous_velocity[:] = 0.
    assert sum(item["history_active"] for item in diagnostics) > 20
    np.testing.assert_array_equal(retargeter.model.jnt_range, ranges)
    retargeter.step(synthesize_human_27_pose(), 4.)
    assert retargeter.experiment_diagnostics["reset_reason"] == "timestamp_gap"
    with pytest.raises(ValueError):
        retargeter.step(synthesize_human_27_pose(), 4.)


def test_structure_really_replaces_point_tasks_and_does_not_add_rotation_measurements():
    retargeter = StructuredHistoryRetargeter(CONFIG)
    points = synthesize_human_27_pose()
    for index in range(45):
        result = retargeter.step(points, index / 30.)
    assert result.observed
    assert len(retargeter.experiment_diagnostics["direction_error_vs_input_deg"]) == 8
    assert not retargeter.t_elbow["left"].cost.any()
    assert not retargeter.t_hand["right"].cost.any()
    assert not retargeter.t_knee["left"].cost.any()
