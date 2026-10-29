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
from g1_imitation.core.s6_mujoco_closed_loop import (  # noqa: E402
    MuJoCoClosedLoopVerifier,
    G1PhysicsPlant,
    SonicPolicy,
)


@pytest.fixture(scope="module")
def policy():
    return SonicPolicy()


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
