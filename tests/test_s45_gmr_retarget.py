"""Module S4+S5 (GMR default retargeter): provenance, failure behaviour, arm geometry, world-convention stability."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from g1_imitation.core.constants import KEYPOINT_INDEX_BY_NAME as K  # noqa: E402
from g1_imitation.core.constants import load_config  # noqa: E402
from g1_imitation.core.s45_gmr_retarget import GMRRetargeter, finite_difference_velocity  # noqa: E402
from g1_imitation.core.s6_mujoco_closed_loop import G1PhysicsPlant  # noqa: E402
from g1_imitation.core.synthetic_human import synthesize_human_27_pose  # noqa: E402


@pytest.fixture(scope="module")
def gmr():
    return GMRRetargeter()


def _seq(n=30, **kw):
    return np.array([synthesize_human_27_pose(root_xy_m=np.array([0.01 * i, 0.0]), **kw) for i in range(n)])


def test_config_pins_third_party_code_and_model_hash():
    cfg = load_config()["s45_gmr_retarget"]
    assert "@25f56db79ac411258b377b6d4ccba0c8ff3372ac" in cfg["gmr_commit"]
    assert "@e6006fd3814d5f8ad45a7ce965beb9bcc90767f2" in cfg["easymocap_commit"]
    assert "@1265df7ba545e8b00f72e7c557c766e15c71632f" in cfg["smplx_python_package"]
    f = Path(cfg["smplx_model_dir"]) / "SMPLX_NEUTRAL.npz"
    assert hashlib.sha256(f.read_bytes()).hexdigest() == cfg["smplx_model_sha256"]["SMPLX_NEUTRAL.npz"]
    assert "s4_se3_scaling" not in load_config() and "s5_mink_ik" not in load_config()


def test_bad_input_raises(gmr):
    with pytest.raises(ValueError):
        gmr.retarget(np.zeros((10, 17, 3)), 30.0)
    with pytest.raises(ValueError):
        gmr.retarget(_seq(5), 0.0)
    with pytest.raises(ValueError):
        gmr.retarget(np.full((10, 27, 3), np.nan), 30.0)


def test_output_layout_limits_and_velocity(gmr):
    res = gmr.retarget(_seq(30, squat_depth_m=0.1), 30.0)
    assert res.q.shape == (30, 36) and res.dq.shape == (30, 35) and res.observed.all()
    assert np.isfinite(res.q).all() and np.allclose(np.linalg.norm(res.q[:, 3:7], axis=1), 1.0, atol=1e-6)
    model = G1PhysicsPlant().kin_model
    lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
    assert np.all(res.q[:, 7:] >= lo - 1e-3) and np.all(res.q[:, 7:] <= hi + 1e-3)
    assert np.allclose(res.dq, finite_difference_velocity(res.q, 1 / 30.0))


def test_t_pose_arms_point_sideways(gmr):
    """The removed S4/S5 folded the arms in front of the chest for a T-pose (~90 deg off). 30 deg separates the two."""
    X = _seq(20, t_pose=True)
    q = gmr.retarget(X, 30.0).q[10]
    plant = G1PhysicsPlant()
    d = mujoco.MjData(plant.kin_model)
    d.qpos[:] = q
    mujoco.mj_kinematics(plant.kin_model, d)

    def body(n):
        return d.xpos[mujoco.mj_name2id(plant.kin_model, mujoco.mjtObj.mjOBJ_BODY, n)]

    def ang(a, b):
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        return np.degrees(np.arccos(np.clip(a @ b, -1, 1)))

    h = X[10]
    for s in ("left", "right"):
        assert ang(body(f"{s}_elbow_link") - body(f"{s}_shoulder_pitch_link"), h[K[f"{s}_elbow"]] - h[K[f"{s}_shoulder"]]) < 30.0
        assert ang(body(f"{s}_wrist_roll_link") - body(f"{s}_elbow_link"), h[K[f"{s}_wrist"]] - h[K[f"{s}_elbow"]]) < 30.0


def test_world_yaw_and_origin_do_not_flip_branches(gmr):
    """Not exactly invariant: EasyMocap's L-BFGS fit (ftol 1e-4, global rotation initialised at 0) lands on slightly
    different optima; measured max 2.9 deg on this sequence (2026-10-28). 5 deg guards against branch flips (the
    removed S5 flipped an arm by ~180 deg under a different world convention)."""
    X = np.array([synthesize_human_27_pose(torso_yaw_rad=0.3 * np.sin(i / 5), l_arm_dir=np.array([0.3 * np.sin(i / 4), 0.4, -0.8]),
                                           root_xy_m=np.array([0.01 * i, 0.0])) for i in range(30)])
    a = np.radians(130.0)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    q1 = gmr.retarget(X, 30.0).q
    q2 = gmr.retarget(X @ R.T + np.array([3.0, -2.0, 0.0]), 30.0).q
    assert np.degrees(np.abs(q1[:, 7:] - q2[:, 7:])).max() < 5.0
