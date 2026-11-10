"""Live Hikrobot 4-camera capture -> S1 2D keypoints -> Undistort & S2 3D skeleton -> S45c G1 retargeting visualization.

Input : `outputs/calibration/physical_cameras_bin2_812x620.json` and 4 live Hikrobot GigE cameras (`812x620` Binning 2).
Output: `outputs/hik_live_pose/live_pose_result.jpg` and `outputs/hik_live_pose/live_pose_report.json`.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from g1_imitation.core.s1_vision_tracker import RTMW27VisionTracker  # noqa: E402
from g1_imitation.core.s1b_person_association import MultiPersonTargetTracker, extract_person_candidates  # noqa: E402
from g1_imitation.core.s2_triangulation import MultiViewTriangulator, undistort_keypoints_2d  # noqa: E402
from g1_imitation.core.s3_temporal_filter import TemporalFilter  # noqa: E402
from g1_imitation.core.s45c_keypoint_ik import KeypointIKRetargeter, retarget_sequence  # noqa: E402
from g1_imitation.core.s6_mujoco_closed_loop import G1PhysicsPlant  # noqa: E402
from hik_camera_check import grab  # noqa: E402
from hik_sync_capture_test import Camera, MvCamera, check, enumerate_devices  # noqa: E402

EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (15, 19), (16, 20), (16, 22),
]


def render_3d_skeleton_panel(pts_3d: np.ndarray, width: int = 540, height: int = 480) -> np.ndarray:
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    valid = np.all(np.isfinite(pts_3d), axis=1)
    if not np.any(valid):
        cv2.putText(canvas, "No valid 3D points", (80, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 200), 2)
        return canvas

    center = np.nanmean(pts_3d[[5, 6, 11, 12]], axis=0)
    if not np.all(np.isfinite(center)):
        center = np.nanmean(pts_3d[valid], axis=0)
    center = np.array([center[0], center[1], 0.85], dtype=np.float64)

    az, el = np.radians(40.0), np.radians(-16.0)
    fwd = np.array([-np.cos(el) * np.cos(az), -np.cos(el) * np.sin(az), np.sin(el)])
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)

    def proj(p: np.ndarray) -> tuple[int, int]:
        d = p - center
        x = float(np.dot(d, right))
        y = float(np.dot(d, up))
        u = int(round(width * 0.5 + x * 240.0))
        v = int(round(height * 0.56 - y * 240.0))
        return u, v

    # Draw Z=0 floor grid around human
    for gx in np.linspace(center[0] - 0.8, center[0] + 0.8, 5):
        cv2.line(canvas, proj(np.array([gx, center[1] - 0.8, 0.0])), proj(np.array([gx, center[1] + 0.8, 0.0])), (210, 210, 210), 1)
    for gy in np.linspace(center[1] - 0.8, center[1] + 0.8, 5):
        cv2.line(canvas, proj(np.array([center[0] - 0.8, gy, 0.0])), proj(np.array([center[0] + 0.8, gy, 0.0])), (210, 210, 210), 1)

    for a, b in EDGES:
        if valid[a] and valid[b]:
            color = (220, 90, 20) if a in (5, 7, 11, 13, 15) else (20, 140, 230)
            cv2.line(canvas, proj(pts_3d[a]), proj(pts_3d[b]), color, 3, cv2.LINE_AA)
    for j in range(27):
        if valid[j]:
            cv2.circle(canvas, proj(pts_3d[j]), 5, (30, 30, 30), -1, cv2.LINE_AA)

    pelvis_z = float(np.nanmean(pts_3d[[11, 12], 2]))
    foot_z = float(np.nanmin(pts_3d[[15, 16, 17, 18, 19, 20, 21, 22], 2]))
    cv2.putText(canvas, "3D Skeleton (World Z-up, m)", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2)
    cv2.putText(canvas, f"Pelvis Z={pelvis_z:.2f}m | Lowest Foot Z={foot_z:.3f}m", (16, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 2)
    return canvas


def main() -> None:
    calib_path = ROOT / "outputs/calibration/physical_cameras_bin2_812x620.json"
    calib = json.loads(calib_path.read_text(encoding="utf-8"))["cameras"]
    serials = sorted(calib.keys())
    K_mats = np.stack([np.array(calib[s]["K"], dtype=np.float64) for s in serials])
    dist_coeffs = np.stack([np.array(calib[s]["dist_coeffs"], dtype=np.float64) for s in serials])
    P_mats = np.stack([np.array(calib[s]["P"], dtype=np.float64) for s in serials])

    out_dir = ROOT / "outputs/hik_live_pose"
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = RTMW27VisionTracker(device="cpu", use_detector_every_frame=True)
    assoc = MultiPersonTargetTracker()
    tri = MultiViewTriangulator(c_min=0.30, reproj_outlier_threshold_px=15.0)
    flt = TemporalFilter(freq_hz=15.0, mode="kalman_3x3")

    MvCamera.MV_CC_Initialize()
    cams = []
    frames_seq = []
    try:
        cams = sorted((Camera(info) for info in enumerate_devices()), key=lambda c: c.serial)
        for i, c in enumerate(cams):
            c.configure(i, 2, 15.0, 6000.0, "freerun", 1500)
            check(c.cam.MV_CC_StartGrabbing(), "start")
        time.sleep(0.3)
        # Grab 20 consecutive frames (~1.3 s) so S1b confirms the target and S45c finishes 0.5 s scale calibration
        for k in range(20):
            imgs = []
            for c in cams:
                bgr, _ = grab(c.cam)
                imgs.append(bgr)
            frames_seq.append((k / 15.0, imgs))
    finally:
        for c in cams:
            c.close()
        MvCamera.MV_CC_Finalize()

    pts_3d_seq = []
    last_kp2d = None
    last_reproj_err = None
    for t_s, imgs in frames_seq:
        cands = extract_person_candidates(tracker, imgs)
        # Undistort candidate 2D keypoints before multi-view epipolar grouping in S1b
        for c_idx, cand in enumerate(cands):
            if len(cand.kp) > 0:
                cand.kp[:] = undistort_keypoints_2d(
                    cand.kp, K_mats[c_idx:c_idx + 1].repeat(len(cand.kp), 0),
                    dist_coeffs[c_idx:c_idx + 1].repeat(len(cand.kp), 0),
                )
        res_s1b = assoc.step(cands, P_mats, t_s)
        kp_used = res_s1b.keypoints_2d
        if not res_s1b.valid:
            # Fallback to single-person detector if S1b is still in its 3-frame birth confirmation window
            det_res = tracker.process_frame(imgs, P_mats, dt=1.0 / 15.0)
            kp_used = undistort_keypoints_2d(det_res.keypoints_2d, K_mats, dist_coeffs)
        r2 = tri.process_frame(kp_used, P_mats, dt_s=1.0 / 15.0)
        r3 = flt.process_frame(r2.points_3d_m, timestamp_s=t_s, covariance_m2=r2.point_covariances_m2)
        pts_3d_seq.append(r3.points_3d_smooth_m.copy())
        last_kp2d = kp_used.copy()
        last_reproj_err = r2.reprojection_errors_px.copy()

    pts_arr = np.asarray(pts_3d_seq, dtype=np.float64)
    t_arr = np.array([f[0] for f in frames_seq], dtype=np.float64)
    rt = KeypointIKRetargeter()
    ik_out = retarget_sequence(pts_arr, t_arr, rt)
    q_last = ik_out["q"][-1]
    pts_last = pts_arr[-1]
    imgs_last = frames_seq[-1][1]
    assert last_kp2d is not None and last_reproj_err is not None

    # Render 4 camera views with detected 2D skeleton + 3D reprojected skeleton
    cam_tiles = []
    Xh = np.c_[np.nan_to_num(pts_last), np.ones(27)]
    for v, (s, im) in enumerate(zip(serials, imgs_last)):
        vis = im.copy()
        uvw = (P_mats[v] @ Xh.T).T
        uv_reproj = uvw[:, :2] / np.maximum(uvw[:, 2:3], 0.1)
        for a, b in EDGES:
            if last_kp2d[v, a, 2] > 0.30 and last_kp2d[v, b, 2] > 0.30:
                cv2.line(vis, tuple(last_kp2d[v, a, :2].astype(int)), tuple(last_kp2d[v, b, :2].astype(int)), (0, 255, 255), 2, cv2.LINE_AA)
            if np.all(np.isfinite(pts_last[[a, b]])):
                cv2.line(vis, tuple(uv_reproj[a].astype(int)), tuple(uv_reproj[b].astype(int)), (0, 80, 255), 2, cv2.LINE_AA)
        for j in range(27):
            if last_kp2d[v, j, 2] > 0.30:
                cv2.circle(vis, tuple(last_kp2d[v, j, :2].astype(int)), 4, (0, 255, 0), -1)
        cv2.putText(vis, f"{s} (yellow=2D, red=3D reproj)", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cam_tiles.append(cv2.resize(vis, (480, 360)))

    left_grid = np.vstack([np.hstack(cam_tiles[:2]), np.hstack(cam_tiles[2:])])  # (720, 960, 3)
    skel_panel = render_3d_skeleton_panel(pts_last, width=540, height=360)

    plant = G1PhysicsPlant()
    plant.set_qpos36(q_last)
    mujoco.mj_forward(plant.model, plant.data)
    cam_mj = mujoco.MjvCamera()
    cam_mj.azimuth, cam_mj.elevation, cam_mj.distance = 140.0, -15.0, 2.6
    cam_mj.lookat[:] = [float(q_last[0]), float(q_last[1]), 0.80]
    renderer = mujoco.Renderer(plant.model, width=540, height=360)
    renderer.update_scene(plant.data, camera=cam_mj)
    g1_panel = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
    renderer.close()
    cv2.putText(g1_panel, "Unitree G1 Retargeted Pose (S45c IK)", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)

    right_col = np.vstack([skel_panel, g1_panel])  # (720, 540, 3)
    full_sheet = np.hstack([left_grid, right_col])  # (720, 1500, 3)
    out_img = out_dir / "live_pose_result.jpg"
    cv2.imwrite(str(out_img), full_sheet)

    valid_3d = int(np.sum(np.all(np.isfinite(pts_last), axis=1)))
    body_reproj = last_reproj_err[5:17]
    med_reproj = float(np.nanmedian(body_reproj)) if np.any(np.isfinite(body_reproj)) else None
    report = {
        "valid_3d_keypoints": f"{valid_3d}/27",
        "median_body_reprojection_err_px": med_reproj,
        "human_pelvis_xyz_m": np.round(np.nanmean(pts_last[[11, 12]], axis=0), 3).tolist(),
        "human_lowest_foot_z_m": float(np.round(np.nanmin(pts_last[15:23, 2]), 3)),
        "g1_root_xyz_m": np.round(q_last[:3], 3).tolist(),
        "ik_solve_ms_median": float(np.median(ik_out["solve_ms"])),
        "output_image": str(out_img),
    }
    (out_dir / "live_pose_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
