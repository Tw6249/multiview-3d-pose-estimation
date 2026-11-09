"""Causal keypoint-IK retargeter (core/s45c_keypoint_ik.py): causality, calibration, geometry, missing data, flight."""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1_imitation.core.constants import (  # noqa: E402
    G1_29_JOINTS,
    KEYPOINT_INDEX_BY_NAME as K,
    ROBAN_21_JOINTS,
)
from g1_imitation.core.s45c_keypoint_ik import (  # noqa: E402
    KeypointIKRetargeter,
    export_heft_deploy_csv,
    load_heft_deploy_csv,
    retarget_sequence,
)
from g1_imitation.core.s6_mujoco_closed_loop import (  # noqa: E402
    generate_10_diverse_human_motions, generate_jump_motions)
from g1_imitation.core.synthetic_human import synthesize_human_27_pose  # noqa: E402

FPS = 30.0


@pytest.fixture(scope="module")
def rt():
    return KeypointIKRetargeter()


@pytest.fixture(scope="module")
def rt_roban():
    return KeypointIKRetargeter(robot="roban_s17")


def _still(n=45, **kw):
    return np.repeat(synthesize_human_27_pose(**kw)[None], n, 0)


def _body(rt, q, name):
    d = mujoco.MjData(rt.model)
    d.qpos[:] = q
    mujoco.mj_kinematics(rt.model, d)
    return d.xpos[rt.model.body(name).id].copy()


def test_prefix_is_independent_of_future_frames(rt):
    walk = generate_10_diverse_human_motions(duration_s=4.0, fps_hz=100.0)["01_forward_walking_gait"][::3]
    t = np.arange(len(walk)) / FPS
    full = retarget_sequence(walk, t, rt)["q"]
    part = retarget_sequence(walk[:70], t[:70], rt)["q"]
    assert np.array_equal(full[:70], part)


def test_calibration_then_startup_ramp(rt):
    out = retarget_sequence(_still(60), np.arange(60) / FPS, rt)
    n_cal = int(np.ceil(rt.cfg["calibration_s"] * FPS)) + 1
    assert not out["observed"][: n_cal - 1].any() and out["observed"][-1]
    assert np.allclose(out["q"][0, 7:], rt.default_q) and np.all(out["dq"][~out["observed"]] == 0)
    assert any("blend" in h for h in out["held"]) and "blend" not in out["held"][-1]


def test_t_pose_arms_sideways_and_straight_legs(rt):
    q = retarget_sequence(_still(60, t_pose=True), np.arange(60) / FPS, rt)["q"][-1]
    for s, sign in (("left", 1.0), ("right", -1.0)):
        upper = _body(rt, q, f"{s}_wrist_yaw_link") - _body(rt, q, f"{s}_shoulder_roll_link")
        ang = np.degrees(np.arccos(np.dot(upper / np.linalg.norm(upper), [0.0, sign, 0.0])))
        assert ang < 15.0, (s, ang)
        shank = _body(rt, q, f"{s}_ankle_pitch_link") - _body(rt, q, f"{s}_knee_link")
        assert np.degrees(np.arccos(-shank[2] / np.linalg.norm(shank))) < 3.0


def test_squat_bends_knees_and_lowers_pelvis(rt):
    stand = retarget_sequence(_still(60), np.arange(60) / FPS, rt)["q"][-1]
    squat = retarget_sequence(_still(60, squat_depth_m=0.25), np.arange(60) / FPS, rt)["q"][-1]
    k = 7 + G1_29_JOINTS.index("left_knee_joint")
    assert squat[k] > stand[k] + 0.4 and squat[2] < stand[2] - 0.1


def test_missing_limb_is_held_and_missing_trunk_holds_everything(rt):
    P = _still(60)
    P[50:, K["left_elbow"]] = np.nan
    P[55:, K["left_hip"]] = np.nan
    out = retarget_sequence(P, np.arange(60) / FPS, rt)
    assert "left_upper" in out["held"][52] and out["observed"][52]
    assert not out["observed"][57] and np.array_equal(out["q"][57], out["q"][54]) and np.all(out["dq"][57] == 0)


def _flip_hands(rt):
    # swapping index/pinky MCP turns the hand frame by ~180 deg (the S2-raw failure on Take_012)
    P = _still(60, t_pose=True)
    Q = P.copy()
    for s in ("left", "right"):
        Q[45:, [K[f"{s}_index_mcp"], K[f"{s}_pinky_mcp"]]] = P[45:, [K[f"{s}_pinky_mcp"], K[f"{s}_index_mcp"]]]
    a, b = (retarget_sequence(x, np.arange(60) / FPS, rt)["q"][-1] for x in (P, Q))
    arm = [7 + G1_29_JOINTS.index(f"{s}_{j}_joint") for s in ("left", "right")
           for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")]
    wrist = [7 + G1_29_JOINTS.index(f"{s}_wrist_{j}_joint") for s in ("left", "right") for j in ("roll", "pitch", "yaw")]
    return np.degrees(np.abs(a[arm] - b[arm])).max(), np.degrees(np.abs(a[wrist] - b[wrist])).max()


def test_default_ignores_hand_orientation(rt):
    assert rt.cfg["wrist_iterations"] == 0
    assert max(_flip_hands(rt)) < 1e-6


def test_optional_wrist_solve_moves_only_the_wrist(rt):
    arm, wrist = _flip_hands(KeypointIKRetargeter(cfg={**rt.cfg, "wrist_iterations": 5}))
    # measured 2026-10-29: <= 2.2 deg (wrist offsets shift the wrist-position task); the former whole-arm hand task: 90 deg
    assert arm < 5.0 and wrist > 10.0


def test_joint_limits_and_flight_clearance(rt):
    jump = generate_jump_motions(duration_s=6.0, fps_hz=100.0)["11_vertical_jump_series"][::3]
    out = retarget_sequence(jump, np.arange(len(jump)) / FPS, rt)
    q = out["q"][out["observed"]]
    assert np.all(q[:, 7:] >= rt.lo - 1e-4) and np.all(q[:, 7:] <= rt.hi + 1e-4)
    human_clear = np.min(jump[out["observed"]][:, 17:23, 2], axis=1)
    d = mujoco.MjData(rt.model)
    sole = []
    for x in q:
        d.qpos[:] = x
        mujoco.mj_kinematics(rt.model, d)
        sole.append(min((d.xpos[b] + rt.sole_local @ d.xmat[b].reshape(3, 3).T)[:, 2].min()
                        for b in (rt.model.body("left_ankle_roll_link").id, rt.model.body("right_ankle_roll_link").id)))
    air = human_clear > 0.05
    ramp = np.array(["blend" in h for h, o in zip(out["held"], out["observed"]) if o])
    err = np.abs(np.array(sole) - human_clear)[~ramp]
    assert air[~ramp].any() and err.max() < 5e-3          # lowest sole point = human sole clearance (flight kept)


def test_optional_ik_branch_search_reports_selected_candidate(rt):
    cfg = {**rt.cfg, "ik_branch_candidates": 5}
    branch_rt = KeypointIKRetargeter(model=rt.model, cfg=cfg)
    pose = _still(60, t_pose=True)
    out = retarget_sequence(pose, np.arange(60) / FPS, branch_rt)
    observed = np.flatnonzero(out["observed"])
    assert len(observed) > 0
    i = int(observed[-1])
    row = branch_rt.step(pose[i], float(i / FPS) + 1.0)
    assert row.ik_candidates_evaluated == 5
    assert row.ik_branch in ("warm_start", "left_plus", "left_minus", "right_plus", "right_minus")
    assert np.isfinite(row.ik_task_cost) and np.isfinite(row.ik_continuity_cost)
    assert np.all(row.q[7:] >= branch_rt.lo - 1e-4) and np.all(row.q[7:] <= branch_rt.hi + 1e-4)


def test_pelvis_pitch_orientation_axis_can_be_weakened_independently(rt):
    cfg = {**rt.cfg, "pelvis_orientation_axis_scale": [1.0, 0.1, 1.0]}
    weak_pitch_rt = KeypointIKRetargeter(model=rt.model, cfg=cfg)
    assert np.allclose(weak_pitch_rt.t_pelvis.cost[:3], rt.t_pelvis.cost[:3])
    assert weak_pitch_rt.t_pelvis.cost[3] == pytest.approx(rt.t_pelvis.cost[3])
    assert weak_pitch_rt.t_pelvis.cost[4] == pytest.approx(rt.t_pelvis.cost[4] * 0.1)
    assert weak_pitch_rt.t_pelvis.cost[5] == pytest.approx(rt.t_pelvis.cost[5])
    assert np.allclose(weak_pitch_rt.t_torso.cost, rt.t_torso.cost)


def test_bad_inputs_raise(rt):
    rt.reset()
    with pytest.raises(ValueError):
        rt.step(np.zeros((26, 3)), 0.0)
    rt.step(_still(1)[0], 0.0)
    with pytest.raises(ValueError):
        rt.step(_still(1)[0], 0.0)


def test_roban_s17_causality_geometry_squat_missing_and_flight(rt_roban, tmp_path):
    walk = generate_10_diverse_human_motions(duration_s=4.0, fps_hz=100.0)["01_forward_walking_gait"][::3]
    t = np.arange(len(walk)) / FPS
    full = retarget_sequence(walk, t, rt_roban)
    part = retarget_sequence(walk[:70], t[:70], rt_roban)["q"]
    assert full["q"].shape == (len(walk), 28) and full["dq"].shape == (len(walk), 27)
    assert np.array_equal(full["q"][:70], part)

    # T-pose geometry
    q_tpose = retarget_sequence(_still(60, t_pose=True), np.arange(60) / FPS, rt_roban)["q"][-1]
    for s, p, sign in (("left", "l", 1.0), ("right", "r", -1.0)):
        upper = _body(rt_roban, q_tpose, f"zarm_{p}4_link_front") - _body(rt_roban, q_tpose, f"zarm_{p}1_link")
        ang = np.degrees(np.arccos(np.dot(upper / np.linalg.norm(upper), [0.0, sign, 0.0])))
        assert ang < 15.0, (s, ang)
        shank = _body(rt_roban, q_tpose, f"leg_{p}5_link") - _body(rt_roban, q_tpose, f"leg_{p}4_link")
        assert np.degrees(np.arccos(-shank[2] / np.linalg.norm(shank))) < 3.0

    # Squat bends knees and lowers base_link
    stand = retarget_sequence(_still(60), np.arange(60) / FPS, rt_roban)["q"][-1]
    squat = retarget_sequence(_still(60, squat_depth_m=0.25), np.arange(60) / FPS, rt_roban)["q"][-1]
    k = 7 + ROBAN_21_JOINTS.index("leg_l4_joint")
    assert squat[k] > stand[k] + 0.4 and squat[2] < stand[2] - 0.1

    # Missing limb and missing trunk
    P = _still(60)
    P[50:, K["left_elbow"]] = np.nan
    P[55:, K["left_hip"]] = np.nan
    out_m = retarget_sequence(P, np.arange(60) / FPS, rt_roban)
    assert "left_upper" in out_m["held"][52] and out_m["observed"][52]
    assert not out_m["observed"][57] and np.array_equal(out_m["q"][57], out_m["q"][54]) and np.all(out_m["dq"][57] == 0)

    # Joint limits and flight clearance on vertical jump
    jump = generate_jump_motions(duration_s=6.0, fps_hz=100.0)["11_vertical_jump_series"][::3]
    out_j = retarget_sequence(jump, np.arange(len(jump)) / FPS, rt_roban)
    q_j = out_j["q"][out_j["observed"]]
    assert np.all(q_j[:, 7:] >= rt_roban.lo - 1e-4) and np.all(q_j[:, 7:] <= rt_roban.hi + 1e-4)
    human_clear = np.min(jump[out_j["observed"]][:, 17:23, 2], axis=1)
    d = mujoco.MjData(rt_roban.model)
    sole = []
    for x in q_j:
        d.qpos[:] = x
        mujoco.mj_kinematics(rt_roban.model, d)
        sole.append(min((d.xpos[b] + rt_roban.sole_local @ d.xmat[b].reshape(3, 3).T)[:, 2].min()
                        for b in (rt_roban.model.body("leg_l6_link").id, rt_roban.model.body("leg_r6_link").id)))
    air = human_clear > 0.05
    ramp = np.array(["blend" in h for h, o in zip(out_j["held"], out_j["observed"]) if o])
    err = np.abs(np.array(sole) - human_clear)[~ramp]
    assert air[~ramp].any() and err.max() < 5e-3

    # 49-column HEFT deploy CSV round-trip
    csv_file = export_heft_deploy_csv(tmp_path / "roban_test.csv", full["q"], full["dq"])
    q_loaded, dq_loaded = load_heft_deploy_csv(csv_file)
    assert np.allclose(q_loaded, full["q"], atol=1e-6)
    assert np.allclose(dq_loaded[:, 6:], full["dq"][:, 6:], atol=1e-6)
