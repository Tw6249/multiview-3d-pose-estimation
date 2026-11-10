"""Real-time streaming: 4 Hikrobot cameras -> S1/S2/S3/S45c (async thread) -> 50 Hz SONIC MuJoCo G1 live window.

Press `q` or `ESC` in the live window to exit.
"""
from __future__ import annotations

from ctypes import byref, memset, sizeof, string_at
import json
import os
from pathlib import Path
import sys
import threading
import time

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from g1_imitation.core.adv_eval import yaw  # noqa: E402
from g1_imitation.core.process_scheduling import disable_power_throttling  # noqa: E402
from g1_imitation.core.s1_vision_tracker import RTMW27VisionTracker  # noqa: E402
from g1_imitation.core.s2_triangulation import MultiViewTriangulator, undistort_keypoints_2d  # noqa: E402
from g1_imitation.core.s3_temporal_filter import TemporalFilter  # noqa: E402
from g1_imitation.core.s45c_keypoint_ik import KeypointIKRetargeter  # noqa: E402
from g1_imitation.core.s6_mujoco_closed_loop import (  # noqa: E402
    COMMON_DEFAULT_Q_29,
    DEFAULT_HEIGHT_M,
    G1PhysicsPlant,
    SonicPolicy,
    compute_tilt_deg_from_quat_wxyz,
)
from hik_live_pose_demo import EDGES  # noqa: E402
from hik_sync_capture_test import MV_FRAME_OUT, Camera, MvCamera, check, enumerate_devices  # noqa: E402


def main() -> None:
    disable_power_throttling()
    calib_path = ROOT / "outputs/calibration/physical_cameras_bin2_812x620.json"
    calib = json.loads(calib_path.read_text(encoding="utf-8"))["cameras"]
    serials = sorted(calib.keys())
    K_mats = np.stack([np.array(calib[s]["K"], dtype=np.float64) for s in serials])
    dist_coeffs = np.stack([np.array(calib[s]["dist_coeffs"], dtype=np.float64) for s in serials])
    P_mats = np.stack([np.array(calib[s]["P"], dtype=np.float64) for s in serials])

    MvCamera.MV_CC_Initialize()
    cams = []
    stop = threading.Event()
    try:
        cams = sorted((Camera(info) for info in enumerate_devices()), key=lambda c: c.serial)
        for i, c in enumerate(cams):
            c.configure(i, 2, 20.0, 6000.0, "freerun", 1500)
            check(c.cam.MV_CC_StartGrabbing(), "start")

        latest_imgs = {c.serial: np.zeros((620, 812, 3), np.uint8) for c in cams}

        def cam_worker(c: Camera):
            f = MV_FRAME_OUT()
            while not stop.is_set():
                memset(byref(f), 0, sizeof(f))
                if c.cam.MV_CC_GetImageBuffer(f, 200) == 0 and f.pBufAddr:
                    w, h = int(f.stFrameInfo.nWidth), int(f.stFrameInfo.nHeight)
                    if int(f.stFrameInfo.nLostPacket) == 0 and int(f.stFrameInfo.nFrameLen) >= w * h:
                        raw = np.frombuffer(string_at(f.pBufAddr, w * h), np.uint8).reshape(h, w)
                        latest_imgs[c.serial] = cv2.cvtColor(raw, cv2.COLOR_BAYER_RG2BGR)
                    c.cam.MV_CC_FreeImageBuffer(f)

        for c in cams:
            threading.Thread(target=cam_worker, args=(c,), daemon=True).start()

        state_lock = threading.Lock()
        q_init = np.zeros(36, dtype=np.float64)
        q_init[2], q_init[3], q_init[7:] = DEFAULT_HEIGHT_M, 1.0, COMMON_DEFAULT_Q_29
        shared = {
            "q_ref": q_init.copy(),
            "dq_ref": np.zeros(35, dtype=np.float64),
            "pts_3d": np.full((27, 3), np.nan, dtype=np.float64),
            "kp_2d": np.zeros((len(serials), 27, 3), dtype=np.float64),
            "vision_fps": 0.0,
            "ik_ms": 0.0,
            "observed": False,
            "yaw0_offset": None,
            "xy0_offset": None,
        }

        def vision_ik_worker():
            tracker = RTMW27VisionTracker(device="cpu", use_detector_every_frame=False)
            tri = MultiViewTriangulator(c_min=0.30, reproj_outlier_threshold_px=15.0)
            flt = TemporalFilter(freq_hz=12.0, mode="kalman_3x3")
            rt = KeypointIKRetargeter()
            t0 = time.perf_counter()
            last_t = t0
            step_count = 0
            while not stop.is_set():
                t_loop = time.perf_counter()
                t_s = t_loop - t0
                dt = max(1e-3, t_loop - last_t)
                last_t = t_loop
                imgs = [latest_imgs[s].copy() for s in serials]
                if step_count % 15 == 0:
                    tracker.mode = tracker.DETECT_MODE
                det_res = tracker.process_frame(imgs, P_mats, dt=dt)
                kp_rect = undistort_keypoints_2d(det_res.keypoints_2d, K_mats, dist_coeffs)
                r2 = tri.process_frame(kp_rect, P_mats, dt_s=dt)
                r3 = flt.process_frame(r2.points_3d_m, timestamp_s=t_s, covariance_m2=r2.point_covariances_m2)
                tracker.update_3d_state(r3.points_3d_smooth_m, r3.velocities_ms, dt=dt)
                ik_frame = rt.step(r3.points_3d_smooth_m, t_s)
                q_out = ik_frame.q.copy()
                dq_out = ik_frame.dq.copy()

                with state_lock:
                    if ik_frame.observed and shared["yaw0_offset"] is None:
                        shared["yaw0_offset"] = float(yaw(q_out[3:7]))
                        shared["xy0_offset"] = q_out[:2].copy()
                    if shared["yaw0_offset"] is not None:
                        rz = Rotation.from_euler("z", -shared["yaw0_offset"])
                        q_out[:2] = rz.apply(np.r_[q_out[:2] - shared["xy0_offset"], 0.0])[:2]
                        r_cur = rz * Rotation.from_quat(q_out[[4, 5, 6, 3]])
                        xyzw = r_cur.as_quat()
                        q_out[3:7] = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]
                        dq_out[:3] = rz.apply(dq_out[:3])
                    shared["q_ref"] = q_out
                    shared["dq_ref"] = dq_out
                    shared["pts_3d"] = r3.points_3d_smooth_m.copy()
                    shared["kp_2d"] = kp_rect.copy()
                    shared["ik_ms"] = float(ik_frame.solve_ms)
                    shared["observed"] = bool(ik_frame.observed)
                    shared["vision_fps"] = 1.0 / max(1e-3, time.perf_counter() - t_loop)
                step_count += 1

        threading.Thread(target=vision_ik_worker, daemon=True).start()

        plant = G1PhysicsPlant()
        policy = SonicPolicy()
        for _ in range(25):
            q_cur, dq_cur, quat_cur, omega_cur = plant.state()
            q_des = policy.step(COMMON_DEFAULT_Q_29, np.zeros(29), Rotation.identity(), q_cur, dq_cur, quat_cur, omega_cur)
            plant.step_pd(q_des, policy.kp, policy.kd, control_dt=0.02)

        cam_mj = mujoco.MjvCamera()
        cam_mj.azimuth, cam_mj.elevation, cam_mj.distance = 135.0, -15.0, 2.6
        renderer = mujoco.Renderer(plant.model, width=640, height=480)

        win = "REAL-TIME G1 Whole-Body Imitation (Press Q to exit)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1520, 720)
        try:
            cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass

        print("REALTIME_STREAM_RUNNING", flush=True)
        tick = 0
        next_tick = time.perf_counter()
        while not stop.is_set():
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(max(0.0, next_tick - now))
            next_tick += 0.02  # 50 Hz physics & policy loop
            if time.perf_counter() - next_tick > 0.1:
                next_tick = time.perf_counter()

            with state_lock:
                q_ref = shared["q_ref"].copy()
                dq_ref = shared["dq_ref"].copy()
                pts_3d = shared["pts_3d"].copy()
                kp_2d = shared["kp_2d"].copy()
                v_fps = shared["vision_fps"]
                ik_ms = shared["ik_ms"]
                obs_ok = shared["observed"]

            # Step 50 Hz SONIC closed-loop physics
            q_cur, dq_cur, quat_cur, omega_cur = plant.state()
            rot_ref = Rotation.from_quat(q_ref[[4, 5, 6, 3]])
            q_des = policy.step(q_ref[7:], dq_ref[6:], rot_ref, q_cur, dq_cur, quat_cur, omega_cur)
            plant.step_pd(q_des, policy.kp, policy.kd, control_dt=0.02)

            # Auto-recover if robot ever falls in long interactive session
            qpos_now = plant.qpos_body()
            if qpos_now[2] < 0.45 or compute_tilt_deg_from_quat_wxyz(qpos_now[3:7]) > 45.0:
                plant.reset()
                policy.reset()

            # Render UI at 25 Hz (every 2 control ticks)
            if tick % 2 == 0:
                Xh = np.c_[np.nan_to_num(pts_3d), np.ones(27)]
                tiles = []
                for v, s in enumerate(serials):
                    vis = latest_imgs[s].copy()
                    uvw = (P_mats[v] @ Xh.T).T
                    uv_r = uvw[:, :2] / np.maximum(uvw[:, 2:3], 0.1)
                    for a, b in EDGES:
                        if kp_2d[v, a, 2] > 0.30 and kp_2d[v, b, 2] > 0.30:
                            cv2.line(vis, tuple(kp_2d[v, a, :2].astype(int)), tuple(kp_2d[v, b, :2].astype(int)), (0, 255, 255), 2)
                        if np.all(np.isfinite(pts_3d[[a, b]])):
                            cv2.line(vis, tuple(uv_r[a].astype(int)), tuple(uv_r[b].astype(int)), (0, 80, 255), 2)
                    cv2.putText(vis, s, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
                    tiles.append(cv2.resize(vis, (400, 360)))
                left_cams = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])  # (720, 800, 3)

                cam_mj.lookat[:] = [float(qpos_now[0]), float(qpos_now[1]), 0.80]
                renderer.update_scene(plant.data, camera=cam_mj)
                right_g1 = cv2.resize(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR), (720, 720))
                status_str = "TRACKING" if obs_ok else "CALIBRATING (Stand still 0.5s)"
                cv2.putText(right_g1, f"LIVE G1 (50Hz SONIC) | {status_str}", (18, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 255, 80) if obs_ok else (0, 200, 255), 2)
                cv2.putText(right_g1, f"Vision: {v_fps:.1f} FPS | IK: {ik_ms:.1f} ms | Press Q to stop", (18, 68),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)

                cv2.imshow(win, np.hstack([left_cams, right_g1]))
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
            tick += 1
    finally:
        stop.set()
        time.sleep(0.2)
        cv2.destroyAllWindows()
        for c in cams:
            c.close()
        MvCamera.MV_CC_Finalize()


if __name__ == "__main__":
    main()
