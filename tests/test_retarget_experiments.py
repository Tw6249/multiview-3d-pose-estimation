"""Focused finite-difference and causal tests for the opt-in retarget experiments."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1_imitation.core.retarget_experiments import RelativePositionTask, SkeletonTemporalFilter, constrain_joint_trajectory


def test_relative_position_jacobian_finite_difference():
    import mujoco
    from g1_imitation.core.s45c_keypoint_ik import KeypointIKRetargeter

    rt = KeypointIKRetargeter()
    task = RelativePositionTask(rt.model, "left_wrist_yaw_link", "torso_link", "torso_link")
    rt.configuration.update(np.r_[0.03, -0.02, rt.pelvis_rest_z + 0.01, 1.0, 0.0, 0.0, 0.0, rt.default_q])
    jac = task.compute_jacobian(rt.configuration)
    eps = 1e-6
    fd = np.empty_like(jac)
    for i in range(rt.model.nv):
        tangent = np.zeros(rt.model.nv)
        tangent[i] = eps
        original = rt.configuration.q.copy()
        plus = original.copy()
        mujoco.mj_integratePos(rt.model, plus, tangent, 1.0)
        rt.configuration.update(plus)
        ep = task.compute_error(rt.configuration)
        tangent[i] = -eps
        minus = original.copy()
        mujoco.mj_integratePos(rt.model, minus, tangent, 1.0)
        rt.configuration.update(minus)
        em = task.compute_error(rt.configuration)
        rt.configuration.update(original)
        fd[:, i] = (ep - em) / (2 * eps)
    assert np.allclose(jac, fd, atol=2e-6)


def test_skeleton_filter_preserves_bone_lengths_and_has_no_future_dependency():
    from g1_imitation.core.constants import KEYPOINT_INDEX_BY_NAME as K

    p = np.full((27, 3), np.nan)
    p[[5, 6, 11, 12], :] = [[0, .12, 1.5], [0, -.12, 1.5], [0, .1, .9], [0, -.1, .9]]
    for s, side in (("left", 1), ("right", -1)):
        sh = K[f"{s}_shoulder"]
        p[sh] = p[5 if side == 1 else 6]
        p[K[f"{s}_elbow"]] = p[sh] + [.12, side * .18, -.04]
        p[K[f"{s}_wrist"]] = p[K[f"{s}_elbow"]] + [.12, side * .15, -.04]
        p[K[f"{s}_hip"]] = p[11 if side == 1 else 12]
        p[K[f"{s}_knee"]] = p[K[f"{s}_hip"]] + [.0, 0, -.42]
        p[K[f"{s}_ankle"]] = p[K[f"{s}_knee"]] + [.0, 0, -.41]
        for n in ("heel", "big_toe", "small_toe"):
            p[K[f"{s}_{n}"]] = p[K[f"{s}_ankle"]] + [.05, .02 * side, -.03]
        for n in ("index_mcp", "pinky_mcp"):
            p[K[f"{s}_{n}"]] = p[K[f"{s}_wrist"]] + [.03, .01 * side, 0]
    p[:5] = [0, 0, 1.7]
    cfg = {"calibration_s": .1, "root_cutoff_hz": 3., "direction_cutoff_hz": 3., "direction_beta": .6,
           "derivative_cutoff_hz": 1., "uncertainty_scale_m": .04, "min_confidence": .15}
    a, b = SkeletonTemporalFilter(cfg), SkeletonTemporalFilter(cfg)
    out_a = out_b = None
    prefix_at_six = None
    prefix_snapshot = None
    for k in range(7):
        x = p.copy()
        x[11:13, 0] += k * .01
        out_a = a.step(x, k / 30)
        if k < 6:
            out_b = b.step(x, k / 30)
        if k == 5:
            prefix_at_six = out_a
            prefix_snapshot = out_a.copy()
    assert np.array_equal(prefix_at_six, prefix_snapshot, equal_nan=True)
    # A later frame does not mutate a previously returned prefix result.
    assert np.array_equal(prefix_at_six, out_b, equal_nan=True)
    for s in ("left", "right"):
        for a_name, b_name in (("shoulder", "elbow"), ("elbow", "wrist"), ("hip", "knee"), ("knee", "ankle")):
            src = np.linalg.norm(p[K[f"{s}_{a_name}"]] - p[K[f"{s}_{b_name}"]])
            got = np.linalg.norm(out_a[K[f"{s}_{a_name}"]] - out_a[K[f"{s}_{b_name}"]])
            assert abs(src - got) < .025
    missing = p.copy()
    missing[K["left_elbow"]] = np.nan
    held = a.step(missing, 7 / 30)
    assert np.isnan(held[K["left_elbow"]]).all()
    assert np.isnan(held[K["left_wrist"]]).all()


def test_joint_governor_respects_rates_limits_and_reports_interventions():
    q = np.zeros((41, 36))
    q[:, 7:] = np.linspace(0, 1.2, 41)[:, None]
    lo, hi = np.full(29, -1.0), np.full(29, 1.0)
    out, changed = constrain_joint_trajectory(q, np.arange(41) * .02, lo, hi, 3.0, 20.0)
    vel = np.diff(out[:, 7:], axis=0) / .02
    acc = np.diff(vel, axis=0) / .02
    assert np.all(out[:, 7:] >= lo) and np.all(out[:, 7:] <= hi)
    assert np.max(np.abs(vel)) <= 3.0 + 1e-8
    assert np.max(np.abs(acc)) <= 20.0 + 1e-7
    assert changed.any()


def test_joint_governor_does_not_modify_root():
    q = np.zeros((5, 36))
    q[:, :7] = np.arange(5)[:, None]
    out, _ = constrain_joint_trajectory(q, np.arange(5) * .02, np.full(29, -1.), np.full(29, 1.), 2., 10.)
    assert np.array_equal(out[:, :7], q[:, :7])


def test_experimental_contact_acquires_releases_and_missing_is_unknown():
    import json
    from g1_imitation.core.retarget_experiments import ExperimentalRetargeter
    from g1_imitation.core.s6_mujoco_closed_loop import G1PhysicsPlant
    from g1_imitation.core.synthetic_human import synthesize_human_27_pose
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "g1_imitation/config/retarget_ablation_20261107.json").read_text())
    model = G1PhysicsPlant().kin_model
    rt = ExperimentalRetargeter(cfg, support=True, model=model)
    p = synthesize_human_27_pose()
    p[17:23, 2] = 0.0
    for k in range(60):
        rt.step(p, k / 30)
    assert np.array_equal(rt.contact, [1, 1])
    up = p.copy()
    up[:, 2] += .2
    rt.step(up, 60 / 30)
    assert np.array_equal(rt.contact, [0, 0])
    up[17] = np.nan
    rt.step(up, 61 / 30)
    assert rt.contact[0] == -1


def test_opt_in_without_features_is_identical_to_baseline_and_prefix_causal():
    import json
    from g1_imitation.core.retarget_experiments import ExperimentalRetargeter
    from g1_imitation.core.s45c_keypoint_ik import KeypointIKRetargeter, retarget_sequence
    from g1_imitation.core.s6_mujoco_closed_loop import G1PhysicsPlant
    from g1_imitation.core.synthetic_human import synthesize_human_27_pose
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "g1_imitation/config/retarget_ablation_20261107.json").read_text())
    model = G1PhysicsPlant().kin_model
    points = np.repeat(synthesize_human_27_pose()[None], 60, axis=0)
    t = np.arange(60) / 30
    baseline = retarget_sequence(points, t, KeypointIKRetargeter(model=model))["q"]
    off = retarget_sequence(points, t, ExperimentalRetargeter(cfg, model=model))["q"]
    assert np.array_equal(baseline, off)
    full = retarget_sequence(points, t, ExperimentalRetargeter(cfg, relations=True, support=True, model=model))["q"]
    prefix = retarget_sequence(points[:50], t[:50], ExperimentalRetargeter(cfg, relations=True, support=True, model=model))["q"]
    assert np.array_equal(full[:50], prefix)
