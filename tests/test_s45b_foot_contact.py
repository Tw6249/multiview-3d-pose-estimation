"""Module S4+S5b (contact-aware foot correction) unit tests on synthetic inputs; no GMR subprocess needed."""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from g1_imitation.core.constants import load_config
from g1_imitation.core.s45b_foot_contact import FootContactCorrector, detect_contacts
from g1_imitation.core.s6_mujoco_closed_loop import (
    COMMON_DEFAULT_Q_29, G1PhysicsPlant, generate_jump_motions)
from g1_imitation.core.synthetic_human import synthesize_human_27_pose

DT = 0.01
CFG = load_config()["s45b_foot_contact"]


@pytest.fixture(scope="module")
def model():
    return G1PhysicsPlant().kin_model


@pytest.fixture(scope="module")
def jumps():
    return generate_jump_motions(duration_s=10.0, fps_hz=100.0)


def _standing_q(model) -> np.ndarray:
    d = mujoco.MjData(model)
    q = np.r_[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, COMMON_DEFAULT_Q_29]
    d.qpos[:] = q
    mujoco.mj_kinematics(model, d)
    q[2] -= d.xpos[model.body("left_ankle_roll_link").id][2] - FootContactCorrector(model)._sole_height()
    return q


def test_no_support_while_human_foot_is_in_the_air(jumps) -> None:
    for name, pts in jumps.items():
        state, _ = detect_contacts(pts, DT, CFG)
        for f, sole in enumerate((pts[:, 17:20], pts[:, 20:23])):
            airborne = sole[..., 2].min(1) > 0.02
            assert not np.any(state[airborne, f] == 1), f"{name}: foot {f} marked support while airborne"
        both_air = (pts[:, 17:23, 2].min(1) > 0.02)
        assert both_air.any(), name
        assert np.all(state[both_air] == 0), f"{name}: flight phase not detected as both-swing"


def test_missing_foot_points_are_unknown_and_never_locked(model) -> None:
    pts = np.stack([synthesize_human_27_pose()] * 200)
    pts[80:140, 19] = np.nan                                     # left heel unobserved
    state, _ = detect_contacts(pts, DT, CFG)
    assert np.all(state[80:140, 0] == -1)
    res = FootContactCorrector(model).correct(np.stack([_standing_q(model)] * 200), pts, DT)
    ramp = round(CFG["blend_s"] / DT)
    assert np.all(res.lock_weight[80 + ramp:140, 0] == 0.0)


def test_planted_foot_is_pinned_when_reference_slides(model) -> None:
    T = 300
    q = np.stack([_standing_q(model)] * T)
    q[:, 0] += 0.05 * np.sin(2 * np.pi * 0.5 * np.arange(T) * DT)  # reference root (and feet) slides 5 cm
    pts = np.stack([synthesize_human_27_pose()] * T)            # human stands still
    res = FootContactCorrector(model).correct(q, pts, DT)
    d = mujoco.MjData(model)
    ids = [model.body(f"{s}_ankle_roll_link").id for s in ("left", "right")]
    xy = np.zeros((T, 2, 2))
    for t in range(T):
        d.qpos[:] = res.q[t]
        mujoco.mj_kinematics(model, d)
        xy[t] = [d.xpos[i][:2] for i in ids]
    locked = (res.lock_weight == 1.0).all(1)
    assert locked.sum() > 0.8 * T
    drift = np.abs(xy[locked] - xy[locked][0]).max()
    assert drift < 0.005, f"planted ankle drifts {drift * 1000:.1f} mm"


def test_flight_frames_keep_gmr_reference(model, jumps) -> None:
    pts = jumps["11_vertical_jump_series"]
    T = len(pts)
    lift = pts[:, 17:23, 2].min(1)
    q = np.stack([_standing_q(model)] * T)
    q[:, 2] += lift                                              # reference root follows the human flight
    res = FootContactCorrector(model).correct(q, pts, DT)
    free = (res.lock_weight == 0.0).all(1)
    airborne = lift > 0.02
    assert (free & airborne).sum() > 0.5 * airborne.sum(), "most of the flight must be free of foot locks"
    d = mujoco.MjData(model)
    lid = model.body("left_ankle_roll_link").id
    z_out, z_in = np.zeros(T), np.zeros(T)
    for t in np.flatnonzero(airborne):
        for qq, z in ((res.q[t], z_out), (q[t], z_in)):
            d.qpos[:] = qq
            mujoco.mj_kinematics(model, d)
            z[t] = d.xpos[lid][2]
    assert np.all(z_out[airborne] > z_in[airborne] - 0.01), "release blend must not drag a lifting foot down"
    assert np.allclose(res.q[free], q[free]), "frames with no locked foot must pass the GMR reference unchanged"
    assert res.q[:, 2].max() >= q[:, 2].max() - 1e-9, "pelvis height must not be clamped during flight"
