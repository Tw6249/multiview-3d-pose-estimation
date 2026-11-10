"""Record live 4-camera sequence -> S1/S1b/S2/S3 3D pose -> S45c IK -> S6 SONIC MuJoCo physics -> MP4 & keyframes.

Input : `outputs/calibration/physical_cameras_bin2_812x620.json` and 4 live Hikrobot GigE cameras (`812x620`).
Output:
  - `outputs/hik_live_pose/imitation_demo.mp4`
  - `outputs/hik_live_pose/imitation_keyframes.jpg`
  - `outputs/hik_live_pose/imitation_report.json`
"""
from __future__ import annotations

import argparse
from ctypes import byref, memset, sizeof, string_at
import json
from pathlib import Path
import sys
import threading
import time

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from g1_imitation.core.adv_eval import anchor_heading  # noqa: E402
from g1_imitation.core.constants import load_config  # noqa: E402
from g1_imitation.core.s1_vision_tracker import RTMW27VisionTracker  # noqa: E402
from g1_imitation.core.s1b_person_association import MultiPersonTargetTracker, extract_person_candidates  # noqa: E402
from g1_imitation.core.s2_triangulation import MultiViewTriangulator, undistort_keypoints_2d  # noqa: E402
from g1_imitation.core.s3_temporal_filter import TemporalFilter  # noqa: E402
from g1_imitation.core.s45c_keypoint_ik import KeypointIKRetargeter, causal_upsample, retarget_sequence  # noqa: E402
from g1_imitation.core.s6_mujoco_closed_loop import G1PhysicsPlant, MuJoCoClosedLoopVerifier, SonicPolicy  # noqa: E402
from hik_live_pose_demo import EDGES, render_3d_skeleton_panel  # noqa: E402
from hik_sync_capture_test import MV_FRAME_OUT, Camera, MvCamera, check, enumerate_devices  # noqa: E402


def record_sequence(serials: list[str], duration_s: float = 8.0, fps: float = 15.0) -> list[tuple[float, list[np.ndarray]]]:
    MvCamera.MV_CC_Initialize()
    cams = []
    stop = threading.Event()
    recorded: list[tuple[float, list[np.ndarray]]] = []
    try:
        cams = sorted((Camera(info) for info in enumerate_devices()), key=lambda c: c.serial)
        for i, c in enumerate(cams):
            c.configure(i, 2, fps, 6000.0, "freerun", 1500)
            check(c.cam.MV_CC_StartGrabbing(), "start")

        latest = {c.serial: np.zeros((620, 812, 3), np.uint8) for c in cams}

        def worker(c: Camera):
            f = MV_FRAME_OUT()
            while not stop.is_set():
                memset(byref(f), 0, sizeof(f))
                if c.cam.MV_CC_GetImageBuffer(f, 200) == 0 and f.pBufAddr:
                    w, h = int(f.stFrameInfo.nWidth), int(f.stFrameInfo.nHeight)
                    if int(f.stFrameInfo.nLostPacket) == 0 and int(f.stFrameInfo.nFrameLen) >= w * h:
                        raw = np.frombuffer(string_at(f.pBufAddr, w * h), np.uint8).reshape(h, w)
                        latest[c.serial] = cv2.cvtColor(raw, cv2.COLOR_BAYER_RG2BGR)
                    c.cam.MV_CC_FreeImageBuffer(f)

        for c in cams:
            threading.Thread(target=worker, args=(c,), daemon=True).start()

        win = "Recording Live Action for G1 Imitation"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 960)
        try:
            cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass

        time.sleep(0.5)
        n_frames = int(round(duration_s * fps))
        period = 1.0 / fps
        t0 = time.perf_counter()
        print(f"RECORDING_STARTED ({n_frames} frames @ {fps:.1f} Hz = {duration_s:.1f} s)", flush=True)
        for k in range(n_frames):
            target_t = t0 + k * period
            while time.perf_counter() < target_t:
                time.sleep(max(0.0, min(0.004, target_t - time.perf_counter())))
            t_rel = time.perf_counter() - t0
            imgs = [latest[s].copy() for s in serials]
            recorded.append((t_rel, imgs))
            if k % 3 == 0:
                tiles = []
                for s, im in zip(serials, imgs):
                    v = cv2.resize(im, (480, 360))
                    cv2.putText(v, f"REC {t_rel:.1f}/{duration_s:.1f}s [{s}]", (12, 32),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    tiles.append(v)
                cv2.imshow(win, np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
                cv2.waitKey(1)
        print("RECORDING_FINISHED", len(recorded), flush=True)
    finally:
        stop.set()
        time.sleep(0.2)
        cv2.destroyAllWindows()
        for c in cams:
            c.close()
        MvCamera.MV_CC_Finalize()
    return recorded


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration-s", type=float, default=8.0)
    ap.add_argument("--fps", type=float, default=15.0)
    args = ap.parse_args()

    calib_path = ROOT / "outputs/calibration/physical_cameras_bin2_812x620.json"
    calib = json.loads(calib_path.read_text(encoding="utf-8"))["cameras"]
    serials = sorted(calib.keys())
    K_mats = np.stack([np.array(calib[s]["K"], dtype=np.float64) for s in serials])
    dist_coeffs = np.stack([np.array(calib[s]["dist_coeffs"], dtype=np.float64) for s in serials])
    P_mats = np.stack([np.array(calib[s]["P"], dtype=np.float64) for s in serials])

    out_dir = ROOT / "outputs/hik_live_pose"
    out_dir.mkdir(parents=True, exist_ok=True)

    recorded = record_sequence(serials, duration_s=args.duration_s, fps=args.fps)

    tracker = RTMW27VisionTracker(device="cpu", use_detector_every_frame=False)
    assoc = MultiPersonTargetTracker()
    tri = MultiViewTriangulator(c_min=0.30, reproj_outlier_threshold_px=15.0)
    flt = TemporalFilter(freq_hz=args.fps, mode="kalman_3x3")

    pts_raw_seq, pts_smooth_seq, kp2d_seq, reproj_seq = [], [], [], []
    dt = 1.0 / args.fps
    locked_once = False
    for k, (t_s, imgs) in enumerate(recorded):
        if not locked_once or k % 8 == 0:
            cands = extract_person_candidates(tracker, imgs)
            for c_idx, cand in enumerate(cands):
                if len(cand.kp) > 0:
                    cand.kp[:] = undistort_keypoints_2d(
                        cand.kp, K_mats[c_idx:c_idx + 1].repeat(len(cand.kp), 0),
                        dist_coeffs[c_idx:c_idx + 1].repeat(len(cand.kp), 0),
                    )
            res_s1b = assoc.step(cands, P_mats, t_s)
            if res_s1b.valid:
                kp_used = res_s1b.keypoints_2d
                locked_once = True
            else:
                det_res = tracker.process_frame(imgs, P_mats, dt=dt)
                kp_used = undistort_keypoints_2d(det_res.keypoints_2d, K_mats, dist_coeffs)
        else:
            det_res = tracker.process_frame(imgs, P_mats, dt=dt)
            kp_used = undistort_keypoints_2d(det_res.keypoints_2d, K_mats, dist_coeffs)

        r2 = tri.process_frame(kp_used, P_mats, dt_s=dt)
        r3 = flt.process_frame(r2.points_3d_m, timestamp_s=t_s, covariance_m2=r2.point_covariances_m2)
        tracker.update_3d_state(r3.points_3d_smooth_m, r3.velocities_ms, dt=dt)
        pts_raw_seq.append(r2.points_3d_m.copy())
        pts_smooth_seq.append(r3.points_3d_smooth_m.copy())
        kp2d_seq.append(kp_used.copy())
        reproj_seq.append(r2.reprojection_errors_px.copy())
        if (k + 1) % 20 == 0 or k == len(recorded) - 1:
            print(f"processed {k + 1}/{len(recorded)} frames", flush=True)

    pts_smooth = np.asarray(pts_smooth_seq, dtype=np.float64)
    t_slot = np.array([f[0] for f in recorded], dtype=np.float64)
    rt = KeypointIKRetargeter()
    ik_out = retarget_sequence(pts_smooth, t_slot, rt)
    t50, q_ref, dq_ref, fresh = causal_upsample(t_slot, ik_out["q"], ik_out["dq"], ik_out["observed"], "hold", rt.lo, rt.hi, dt=0.02)
    q_ref, dq_ref, _ = anchor_heading(q_ref, dq_ref)

    phys_lowlat = MuJoCoClosedLoopVerifier().run_stage_6b_physics(
        q_ref, dq_ref, ik_dt=0.02, control_dt=0.02, settle_steps=25, preview_s=0.0
    )

    plant = G1PhysicsPlant()
    cam_mj = mujoco.MjvCamera()
    cam_mj.azimuth, cam_mj.elevation, cam_mj.distance = 135.0, -15.0, 2.7
    renderer = mujoco.Renderer(plant.model, width=480, height=360)

    mp4_path = out_dir / "imitation_demo.mp4"
    writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1440, 720))
    keyframe_rows = []
    key_indices = sorted({min(len(recorded) - 1, int(round(f * (len(recorded) - 1)))) for f in (0.15, 0.40, 0.65, 0.90)})

    try:
        for k, (t_s, imgs) in enumerate(recorded):
            kp = kp2d_seq[k]
            X = pts_smooth[k]
            Xh = np.c_[np.nan_to_num(X), np.ones(27)]
            cam_tiles = []
            for v, (s, im) in enumerate(zip(serials, imgs)):
                vis = im.copy()
                uvw = (P_mats[v] @ Xh.T).T
                uv_r = uvw[:, :2] / np.maximum(uvw[:, 2:3], 0.1)
                for a, b in EDGES:
                    if kp[v, a, 2] > 0.30 and kp[v, b, 2] > 0.30:
                        cv2.line(vis, tuple(kp[v, a, :2].astype(int)), tuple(kp[v, b, :2].astype(int)), (0, 255, 255), 2, cv2.LINE_AA)
                    if np.all(np.isfinite(X[[a, b]])):
                        cv2.line(vis, tuple(uv_r[a].astype(int)), tuple(uv_r[b].astype(int)), (0, 80, 255), 2, cv2.LINE_AA)
                cv2.putText(vis, f"{s} t={t_s:.2f}s", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cam_tiles.append(cv2.resize(vis, (320, 240)))
            cams_col = np.vstack([
                np.hstack(cam_tiles[:2]),
                np.hstack(cam_tiles[2:]),
                render_3d_skeleton_panel(X, width=640, height=240),
            ])  # (720, 640, 3)

            idx50 = int(np.clip(np.searchsorted(t50, t_s + 1e-9, side="right") - 1, 0, len(t50) - 1))
            q_r = q_ref[idx50]
            q_p = phys_lowlat.qpos_hist[idx50]
            cam_mj.lookat[:] = [float(q_r[0]), float(q_r[1]), 0.80]

            plant.set_qpos36(q_r)
            mujoco.mj_forward(plant.model, plant.data)
            renderer.update_scene(plant.data, camera=cam_mj)
            im_ref = cv2.resize(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR), (800, 360))
            cv2.putText(im_ref, "G1 Causal IK Reference (S45c)", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)

            plant.set_qpos36(q_p)
            mujoco.mj_forward(plant.model, plant.data)
            renderer.update_scene(plant.data, camera=cam_mj)
            im_phys = cv2.resize(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR), (800, 360))
            cv2.putText(im_phys, f"G1 Closed-Loop Physics (SONIC low_latency, survived={phys_lowlat.survived})",
                        (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 255, 100), 2)

            frame_canvas = np.hstack([cams_col, np.vstack([im_ref, im_phys])])  # (720, 1440, 3)
            writer.write(frame_canvas)
            if k in key_indices:
                keyframe_rows.append(cv2.resize(frame_canvas, (960, 480)))
    finally:
        writer.release()
        renderer.close()

    if keyframe_rows:
        kf_sheet = np.vstack([np.hstack(keyframe_rows[:2]), np.hstack(keyframe_rows[2:])])
        cv2.imwrite(str(out_dir / "imitation_keyframes.jpg"), kf_sheet)

    reproj_arr = np.asarray(reproj_seq, dtype=np.float64)[:, 5:17]
    report = {
        "frames_recorded": len(recorded),
        "duration_s": float(recorded[-1][0]),
        "valid_3d_frame_ratio": float(np.mean(np.all(np.isfinite(pts_smooth[:, 5:17]), axis=(1, 2)))),
        "median_body_reprojection_px": float(np.nanmedian(reproj_arr)),
        "ik_solve_ms_median": float(np.median(ik_out["solve_ms"][ik_out["observed"]])),
        "closed_loop_survived": bool(phys_lowlat.survived),
        "closed_loop_joint_rmse_rad": float(phys_lowlat.joint_rmse_rad),
        "closed_loop_joint_rmse_deg": float(phys_lowlat.joint_rmse_deg),
        "closed_loop_max_tilt_deg": float(phys_lowlat.max_tilt_deg),
        "closed_loop_min_pelvis_z_m": float(phys_lowlat.min_pelvis_height_m),
        "closed_loop_foot_skating_m_s": float(phys_lowlat.support_foot_skating_speed_ms),
        "video_path": str(mp4_path),
        "keyframes_path": str(out_dir / "imitation_keyframes.jpg"),
    }
    (out_dir / "imitation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
