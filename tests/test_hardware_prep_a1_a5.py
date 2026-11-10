"""Verification tests for Hardware Preparation Tasks A1–A5 (Distortion, R2 Calibration, R4 Safety Governor, R1 PoE Capture, R3 Seqlock/DDS)."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pytest

from g1_imitation.core.r1_hik_capture import (
    SimulatedBayerCaptureBackend,
    assemble_synchronized_bundle,
    build_hik_register_plan,
)
from g1_imitation.core.r3_shared_memory_dds import (
    SeqlockReferenceBuffer,
    UnitreeG1DdsBridge,
)
from g1_imitation.core.r4_safety_governor import SafetyGovernor
from g1_imitation.core.s2_triangulation import MultiViewTriangulator, create_circular_camera_rig, undistort_keypoints_2d
from g1_imitation.core.s45c_keypoint_ik import KeypointIKFrame
from g1_imitation.core.s6_mujoco_closed_loop import COMMON_DEFAULT_Q_29, DEFAULT_HEIGHT_M
from g1_imitation.core.synthetic_human import synthesize_human_27_pose
from scripts.calibrate_physical_cameras import create_charuco_board, run_self_test


def test_a1_radial_distortion_undistort_recovers_sub_10mm_triangulation() -> None:
    """Task A1: Radial-tangential lens distortion compensation restores triangulation accuracy < 10 mm."""
    pts_true = synthesize_human_27_pose(height_m=1.75, root_xy_m=np.array([0.85, 0.65]))
    P_mats, cam_params = create_circular_camera_rig(num_cameras=4, radius_m=2.6, camera_height_m=1.4, focal_px=650.0)
    K_mats = np.stack([c["K"] for c in cam_params], axis=0)
    R_mats = np.stack([c["R_w2c"] for c in cam_params], axis=0)
    t_vecs = np.stack([c["t_w2c"] for c in cam_params], axis=0)
    dist_coeffs = np.tile(np.array([-0.14, 0.035, 0.0005, -0.0004, 0.0], dtype=np.float64), (4, 1))

    kp_distorted = np.zeros((4, 27, 3), dtype=np.float64)
    for c in range(4):
        rvec, _ = cv2.Rodrigues(R_mats[c])
        uv_dist, _ = cv2.projectPoints(pts_true, rvec, t_vecs[c], K_mats[c], dist_coeffs[c])
        kp_distorted[c, :, :2] = uv_dist.reshape(27, 2)
        kp_distorted[c, :, 2] = 0.95

    tri = MultiViewTriangulator(use_consensus_search=False)
    res_uncorrected = tri.process_frame(kp_distorted, P_mats, update_history=False)
    err_uncorrected_mm = float(np.mean(np.linalg.norm(res_uncorrected.points_3d_m - pts_true, axis=1)) * 1e3)

    tri.reset()
    res_corrected = tri.process_frame(
        kp_distorted,
        P_mats,
        update_history=False,
        K_matrices=K_mats,
        dist_coeffs=dist_coeffs,
    )
    err_corrected_mm = float(np.mean(np.linalg.norm(res_corrected.points_3d_m - pts_true, axis=1)) * 1e3)

    assert err_uncorrected_mm > 15.0
    assert err_corrected_mm < 10.0, f"Expected undistorted MPJPE < 10 mm, got {err_corrected_mm:.3f} mm"
    assert np.allclose(undistort_keypoints_2d(kp_distorted, None, None), kp_distorted)


def test_a2_charuco_detector_and_floor_z0_alignment(tmp_path: Path) -> None:
    """Task A2: ChArUco board detection + multi-camera calibration floor Z=0 alignment (< 3.0 mm residual)."""
    board, detector = create_charuco_board()
    board_img = board.generateImage((1100, 800), marginSize=40)
    board_bgr = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)
    corners, ids, _, _ = detector.detectBoard(board_bgr)
    assert corners is not None and ids is not None and len(ids) >= 50

    out_json = tmp_path / "physical_cameras.json"
    calib = run_self_test(out_json)
    assert out_json.exists()
    assert calib["floor_z_max_abs_err_mm"] < 3.0
    assert len(calib["cameras"]) == 4


def test_a3_r4_safety_governor_soft_clutch_fall_envelope_and_timeout() -> None:
    """Task A3: R4 safety governor enforces 2.0s soft clutch, velocity bounds, fall blocking, and timeout return."""
    gov = SafetyGovernor(clutch_duration_s=2.0, short_hold_timeout_s=0.15, return_to_stand_duration_s=1.5)
    pts_stand = synthesize_human_27_pose(height_m=1.75)
    q_active = np.zeros(36, dtype=np.float64)
    q_active[2], q_active[3] = DEFAULT_HEIGHT_M, 1.0
    q_active[7:] = COMMON_DEFAULT_Q_29 + 0.15
    ik_ok = KeypointIKFrame(q=q_active, dq=np.zeros(35, dtype=np.float64), observed=True)

    # 1. Soft clutch over 0..2.0 s
    f0 = gov.step(ik_ok, pts_stand, 0.00)
    assert f0.safety_state == "SOFT_CLUTCH" and f0.fault_reason is None
    for k in range(1, 105):
        fk = gov.step(ik_ok, pts_stand, k * 0.02)
    assert fk.safety_state == "ACTIVE_TRACK"
    assert np.allclose(fk.q_safe[7:], q_active[7:], atol=1e-3)

    # 2. Human fall envelope block (z_pelvis < 0.48 m)
    pts_fallen = pts_stand.copy()
    pts_fallen[:, 2] -= 0.55
    q_fallen = q_active.copy()
    q_fallen[2] = 0.25
    f_fall = gov.step(KeypointIKFrame(q=q_fallen, dq=np.zeros(35), observed=True), pts_fallen, 2.12)
    assert f_fall.safety_state == "HOLD_POSE"
    assert f_fall.fault_reason == "human_out_of_envelope"
    assert float(f_fall.q_safe[2]) >= 0.70  # robot refuses to drop pelvis to 0.25 m

    # 3. Missing observation short hold (<= 0.15 s) then smooth return to stand (> 0.15 s)
    ik_lost = KeypointIKFrame(q=q_active, dq=np.zeros(35), observed=False)
    f_hold = gov.step(ik_lost, pts_stand, 2.20)
    assert f_hold.safety_state == "HOLD_POSE"
    for step in range(1, 85):
        f_ret = gov.step(ik_lost, pts_stand, 2.25 + step * 0.02)
    assert f_ret.safety_state == "RETURN_TO_STAND"
    assert np.allclose(f_ret.q_safe[7:], COMMON_DEFAULT_Q_29, atol=1e-3)


def test_a5_r1_poe_capture_modes_and_skew_rejection() -> None:
    """Task A5 (R1): Pure-PoE register plans (`ptp_action`, `ptp_freerun`, `software`) and skew rejection (> 1.0 ms)."""
    p_action = build_hik_register_plan(2, 4, trigger_mode="ptp_action")
    assert p_action["GevIEEE1588"] is True and p_action["TriggerSource"] == "Action1" and p_action["GevSCPD"] == 2400
    p_free = build_hik_register_plan(0, 4, trigger_mode="ptp_freerun", fps_hz=50.0)
    assert p_free["GevIEEE1588"] is True and p_free["TriggerMode"] == "Off" and p_free["AcquisitionFrameRate"] == 50.0

    sim = SimulatedBayerCaptureBackend(trigger_mode="ptp_action")
    b_ok = sim.grab_next_bundle(injected_skew_ms=np.array([0.0, 0.15, -0.10, 0.05]))
    assert b_ok.bundle_valid and np.all(b_ok.valid_cam_mask)
    assert b_ok.images_bgr.shape == (4, 1240, 1624, 3)

    # Camera 3 skewed by 3.5 ms (> 1.0 ms gate) -> rejected, remaining 3 cameras stay valid
    b_skew = sim.grab_next_bundle(injected_skew_ms=np.array([0.0, 0.1, -0.1, 3.5]))
    assert b_skew.bundle_valid
    assert list(b_skew.valid_cam_mask) == [True, True, True, False]
    assert np.isnan(b_skew.dev_timestamps_s[3]) and not np.any(b_skew.images_bgr[3])

    # 3 cameras dropped -> < 2 valid cameras -> bundle_valid False
    b_drop = sim.grab_next_bundle(dropped_cameras={0, 1, 2})
    assert not b_drop.bundle_valid and not np.any(b_drop.valid_cam_mask)


def test_a5_r3_seqlock_shared_memory_and_watchdog_trip() -> None:
    """Task A5 (R3): Tear-free Seqlock shared memory round-trip and < 35 ms watchdog damping trip."""
    writer = SeqlockReferenceBuffer(create=True, watchdog_timeout_ms=35.0)
    reader = SeqlockReferenceBuffer(name=writer.name, create=False, watchdog_timeout_ms=35.0)
    try:
        q = np.zeros(36, dtype=np.float64)
        q[2], q[3], q[7:] = DEFAULT_HEIGHT_M, 1.0, COMMON_DEFAULT_Q_29 + 0.05
        dq = np.linspace(-0.2, 0.2, 35, dtype=np.float64)
        seq = writer.write(q, dq, safety_state_code=2, t_monotonic_s=100.000)
        assert seq == 2

        snap_fresh = reader.read(now_monotonic_s=100.012)
        assert not snap_fresh.watchdog_expired and snap_fresh.age_ms == pytest.approx(12.0, abs=0.1)
        assert np.allclose(snap_fresh.q_safe, q) and np.allclose(snap_fresh.dq_safe, dq)

        snap_stale = reader.read(now_monotonic_s=100.040)
        assert snap_stale.watchdog_expired and snap_stale.age_ms == pytest.approx(40.0, abs=0.1)

        bridge = UnitreeG1DdsBridge(kp_29=np.full(29, 80.0), kd_29=np.full(29, 4.0))
        cmd_ok = bridge.build_low_cmd(q[7:], snap_fresh)
        assert not cmd_ok.damping_mode and np.allclose(cmd_ok.kp_29, 80.0)

        cmd_trip = bridge.build_low_cmd(q[7:], snap_stale)
        assert cmd_trip.damping_mode and np.allclose(cmd_trip.kp_29, 0.0) and np.all(cmd_trip.kd_29 > 0.0)
    finally:
        reader.close()
        writer.close()
