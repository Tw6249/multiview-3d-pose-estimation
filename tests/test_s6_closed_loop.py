"""Unit & Integration Tests for Module S6: MuJoCo Dual-Stage Closed-Loop Verification (Stage 6A & Stage 6B).

Verification Criteria (`docs/方案规范.md` — S6.3):
  1. Across 10 distinct 10-second (1000-frame at 100 Hz) human motions (including forward walking,
      turning with stepping, deep squatting, waist bending, and cross-morphology 1.90m/1.55m gaits):
      - Closed-loop balance survival rate == 100% (`z_pelvis > 0.45 m`, `tilt < 45 deg`).
      - Closed-loop joint tracking RMSE < 0.12 rad (~6.8 deg).
      - Support foot (`F_z > 20 N`) average horizontal skating speed < 0.04 m/s (4.0 cm/s).
      - Smooth motor torques without divergence.
"""
from __future__ import annotations

import numpy as np
import pytest

from g1_imitation.core.s6_mujoco_closed_loop import (
    MuJoCoClosedLoopVerifier,
    generate_10_diverse_human_motions,
    solve_ik_trajectory_from_3d,
)


@pytest.fixture(scope="module")
def ten_motion_reports():
    motions = generate_10_diverse_human_motions(duration_s=10.0, fps_hz=100.0)
    assert len(motions) == 10
    verifier = MuJoCoClosedLoopVerifier()
    reports = {}
    for name, pts_seq in motions.items():
        # Module S4+S5 = default causal keypoint IK (30 Hz camera slots, newest slot held on the 100 Hz grid; the 0.5 s
        # calibration stance is not part of the reference)
        q_traj, dq_traj, _ = solve_ik_trajectory_from_3d(pts_seq, dt=0.01)
        rep_6a = verifier.run_stage_6a_kinematic(q_traj, dt=0.01)
        rep_6b = verifier.run_stage_6b_physics(q_traj, dq_traj, ik_dt=0.01, control_dt=0.02, settle_steps=25)
        reports[name] = (rep_6a, rep_6b)
    return reports


def test_s6_all_motions_kinematic_limits_survival_and_finite_torque(ten_motion_reports) -> None:
    for name, (rep_6a, rep_6b) in ten_motion_reports.items():
        assert rep_6a.joint_limit_violation_rate == 0.0, f"Stage 6A joint limit violation on {name}"
        assert rep_6b.survived, (
            f"Motion {name} failed balance survival: min_z={rep_6b.min_pelvis_height_m:.3f} m, "
            f"max_tilt={rep_6b.max_tilt_deg:.2f} deg"
        )
        assert np.all(np.isfinite(rep_6b.torques_hist)), f"Non-finite torques in {name}"


_MOTIONS = (
    "01_forward_walking_gait", "02_turning_stepping_gait", "03_deep_squat_cycle", "04_waist_bend_reach",
    "05_lateral_stepping_sway", "06_torso_twist_arm_wave", "07_boxing_guard_combo",
    "08_overhead_bimanual_lift", "09_tall_subject_190cm_walk_squat", "10_short_subject_155cm_walk_turn",
)
# Unmodified spec thresholds (joint RMSE < 0.12 rad, support-foot skating < 0.04 m/s). Re-measured 2026-10-29 on the
# official G1 plant with the default causal keypoint IK + SONIC v1.1 (scripts/s6_ten_motions_measure.py ->
# outputs/advanced_take/s6_ten_motions_sonic_kpik.json): RMSE 0.065-0.100 rad, skating 0.001-0.020 m/s,
# all ten within both criteria (the offline SMPL-X + GMR reference: RMSE 0.122-0.224 rad on all ten and skating
# 0.046-0.098 m/s on four, s6_ten_motions_sonic_gmr.json). The synthetic walks are tracked in place (the robot does
# not step forward), so their low skating is not evidence of walking.
_RMSE_CASES = list(_MOTIONS)
_SKATING_CASES = list(_MOTIONS)


@pytest.mark.parametrize("motion", _RMSE_CASES)
def test_s6_joint_rmse_below_spec_threshold(ten_motion_reports, motion) -> None:
    rep_6b = ten_motion_reports[motion][1]
    assert rep_6b.joint_rmse_rad < 0.12, f"Motion {motion} joint RMSE {rep_6b.joint_rmse_rad:.4f} rad >= 0.12 rad"


@pytest.mark.parametrize("motion", _SKATING_CASES)
def test_s6_skating_per_motion_below_spec_threshold(ten_motion_reports, motion) -> None:
    rep_6b = ten_motion_reports[motion][1]
    assert rep_6b.support_foot_skating_speed_ms < 0.04, (
        f"Motion {motion} support foot skating speed {rep_6b.support_foot_skating_speed_ms:.4f} m/s >= 0.04 m/s"
    )


def test_s6_skating_mean_below_spec_threshold(ten_motion_reports) -> None:
    skating = [rep_6b.support_foot_skating_speed_ms for _, rep_6b in ten_motion_reports.values()]
    assert float(np.mean(skating)) < 0.04


def test_s6_waist_bend_strict_per_motion_skating_bound(ten_motion_reports) -> None:
    rep_6b = ten_motion_reports["04_waist_bend_reach"][1]
    assert rep_6b.support_foot_skating_speed_ms < 0.04, (
        f"04_waist_bend_reach support foot skating speed {rep_6b.support_foot_skating_speed_ms:.4f} m/s >= 0.04 m/s"
    )
