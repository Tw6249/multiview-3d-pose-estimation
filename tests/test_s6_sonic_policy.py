"""SONIC v1.1 policy port: asset fingerprints, failure behaviour, standing.

The step-by-step equivalence test against the old mainline adapter (motion_control.comparison_policies) was
removed with that code on 2026-10-29; the ONNX sha256 pins and gains stay in pipeline_config.json."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from g1_imitation.core.constants import load_config  # noqa: E402
from g1_imitation.core.s45c_keypoint_ik import load_heft_deploy_csv  # noqa: E402
from g1_imitation.core.s6_mujoco_closed_loop import (  # noqa: E402
    G1PhysicsPlant,
    HeftPolicy,
    MuJoCoClosedLoopVerifier,
    RobanPhysicsPlant,
    SonicPolicy,
)


@pytest.fixture(scope="module")
def policy():
    return SonicPolicy()


@pytest.fixture(scope="module")
def heft_policy():
    return HeftPolicy()


def test_asset_fingerprint_mismatch_and_bad_reference_raise(policy):
    bad = copy.deepcopy(load_config()["s6_closed_loop"]["sonic_v11"])
    bad["encoder_onnx_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        SonicPolicy(bad)
    plant = G1PhysicsPlant()
    st = plant.state()
    ref_q = policy.default_q.copy()
    ref_q[3] = np.nan
    with pytest.raises(ValueError):
        policy.step(ref_q, np.zeros(29), Rotation.identity(), *st)
    with pytest.raises(ValueError):
        policy.step(policy.default_q, np.zeros(28), Rotation.identity(), *st)


def test_stands_for_six_seconds_at_its_default_pose():
    v = MuJoCoClosedLoopVerifier()
    T = 300
    q = np.zeros((T, 36))
    q[:, 2], q[:, 3], q[:, 7:] = 0.793, 1.0, v.policy.default_q
    rep = v.run_stage_6b_physics(q, np.zeros((T, 35)), ik_dt=0.02, control_dt=0.02)
    assert rep.survived and rep.max_tilt_deg < 10.0 and rep.min_pelvis_height_m > 0.7
    assert np.isfinite(rep.torques_hist).all()


def test_heft_roban_s17_fingerprint_standing_and_official_walk_clip(heft_policy):
    bad = copy.deepcopy(load_config()["s6_closed_loop"]["heft_roban_s17"])
    bad["policy_onnx_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        HeftPolicy(bad)
    plant = RobanPhysicsPlant()
    st = plant.state()
    ref_q = heft_policy.default_q.copy()
    ref_q[3] = np.nan
    with pytest.raises(ValueError):
        heft_policy.step(ref_q, np.zeros(21), Rotation.identity(), *st)
    with pytest.raises(ValueError):
        heft_policy.step(heft_policy.default_q, np.zeros(20), Rotation.identity(), *st)

    # Stand 6s at default pose
    v = MuJoCoClosedLoopVerifier(robot="roban_s17")
    T = 300
    q = np.zeros((T, 28))
    q[:, 2], q[:, 3], q[:, 7:] = 0.75, 1.0, v.policy.default_q
    rep_stand = v.run_stage_6b_physics(q, np.zeros((T, 27)), ik_dt=0.02, control_dt=0.02)
    assert rep_stand.survived and rep_stand.max_tilt_deg < 10.0 and rep_stand.min_pelvis_height_m > 0.7
    assert np.isfinite(rep_stand.torques_hist).all()

    # Official HEFT walking deploy CSV (50 Hz) -> physical closed-loop forward walking (> 1.4 m XY displacement)
    assets_root = Path(load_config()["runtime"]["robot_assets_root"])
    walk_csv = assets_root / "controller-comparison/heft_roban_s17/motion_data/004_walking_forward_slowly_deploy.csv"
    q_walk, dq_walk = load_heft_deploy_csv(walk_csv)
    rep_walk = v.run_stage_6b_physics(q_walk, dq_walk, ik_dt=0.02, control_dt=0.02, settle_steps=25, preview_s=0.16)
    assert rep_walk.survived and rep_walk.max_tilt_deg < 10.0 and rep_walk.min_pelvis_height_m > 0.7
    assert float(np.linalg.norm(rep_walk.qpos_hist[-1, :2] - rep_walk.qpos_hist[0, :2])) > 1.4


