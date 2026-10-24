"""End-to-end multi-person cross-view epipolar clustering, target locking,
and 4-stage G1 robot imitation showcase on the outdoor MAMMA dataset.

Pipeline stages visualized synchronously in a 4-quadrant split screen:
  1. Multi-Camera Raw Human RGB Video (4 outdoor views: Cam A, B, C, D) with bounding boxes & target locked
  2. 2D RTMPose Keypoints & 3D Triangulated Human Skeleton in world coordinates
  3. Robot Kinematic Retargeting (BRM-C2 with singularity barrier)
  4. Robot Closed-Loop Full-Dynamics Physical Tracking (MuJoCo 1000 Hz physics)
"""
from __future__ import annotations

import json
import math
import pickle
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from pose_sim.runtime import create_body
from pose_sim.landmarks import SCHEMAS
from pose_sim.geometry import triangulate, CameraGeometry
from motion_control.continuous_pose_reference import build_brm_c2_reference_stream
from motion_control.twist import TwistPlant
from scripts.benchmark_controllers import Plant, FACTORIES, packet, window_indices
from scripts.benchmark_twist_policy import balance_violation

SESSION_DIR = ROOT / "outputs/public_mamma_session"
OUTPUT_DIR = ROOT / "outputs/mamma_teleop_demo"
FIG_DIR = OUTPUT_DIR / "figures"

# 23-point skeleton connectivity for 2D/3D visualization
SKELETON_EDGES = [
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"), ("left_ankle", "left_big_toe"),
    ("right_ankle", "right_heel"), ("right_ankle", "right_big_toe"),
]


def load_fonts() -> dict[int, ImageFont.FreeTypeFont]:
    fonts = {}
    win_fonts = ["msyh.ttc", "simhei.ttf", "arial.ttf"]
    for sz in (12, 13, 14, 15, 16, 18, 22):
        f = None
        for wf in win_fonts:
            p = Path("C:/Windows/Fonts") / wf
            if p.exists():
                try:
                    f = ImageFont.truetype(str(p), sz)
                    break
                except Exception:
                    pass
        if f is None:
            f = ImageFont.load_default()
        fonts[sz] = f
    return fonts


def get_camera_ray(cam: dict, pt_px: list[float]) -> tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy = cam["K"][0][0], cam["K"][1][1], cam["K"][0][2], cam["K"][1][2]
    R = np.array(cam["R_world_to_camera"])
    t = np.array(cam["t_world_to_camera_m"])
    C = -R.T @ t
    ray_cam = np.array([(pt_px[0] - cx) / fx, (pt_px[1] - cy) / fy, 1.0])
    ray_cam /= np.linalg.norm(ray_cam)
    ray_world = R.T @ ray_cam
    return C, ray_world


def ray_distance_and_midpoint(c1: np.ndarray, r1: np.ndarray, c2: np.ndarray, r2: np.ndarray):
    A = np.column_stack([r1, -r2])
    b = c2 - c1
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    t1, t2 = sol[0], sol[1]
    if t1 < 0 or t2 < 0:
        return 999.0, None
    p1 = c1 + t1 * r1
    p2 = c2 + t2 * r2
    return float(np.linalg.norm(p1 - p2)), 0.5 * (p1 + p2)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fonts = load_fonts()

    print("=== Step 1: Loading Session & Full 426-Frame Videos ===")
    cameras_cfg = json.loads((SESSION_DIR / "cameras.json").read_text(encoding="utf-8"))["cameras"]
    cids = ["A001", "B001", "C001", "D001"]
    video_dir = ROOT / "outputs/public_mamma"
    test_cap = cv2.VideoCapture(str(video_dir / "A001.mp4"))
    n_total_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    test_cap.release()

    _, names, indices = SCHEMAS["hybrid"]
    name_to_idx = {n: i for i, n in enumerate(names)}
    anchors = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    anchor_indices = [name_to_idx[n] for n in anchors]

    print(f"Loaded full MAMMA sequence: {n_total_frames} frames ({n_total_frames / 30.0:.2f} s @ 30 FPS) for 4 cameras: {cids}")

    print("=== Step 2: Multi-Person 2D Inference & Epipolar-Temporal Clustering ===")
    full_cache_path = OUTPUT_DIR / "cached_full_426_cands.pkl"
    sub_cache_path = OUTPUT_DIR / "cached_target_history.pkl"
    if full_cache_path.exists():
        print(f"  Loading cached 426-frame raw 2D detections from {full_cache_path.name}...")
        all_cands_per_frame = pickle.loads(full_cache_path.read_bytes())
    else:
        sub_cands = {}
        if sub_cache_path.exists():
            raw_sub = pickle.loads(sub_cache_path.read_bytes())
            if len(raw_sub) == 120:
                for idx_sub, entry in enumerate(raw_sub):
                    sub_cands[60 + idx_sub] = entry["all_cands"]
                print(f"  Reusing {len(sub_cands)} cached frames (60..179) from {sub_cache_path.name}")

        body, _ = create_body(mode="balanced", device="cuda", model_kind="hybrid")
        caps = {cid: cv2.VideoCapture(str(video_dir / f"{cid}.mp4")) for cid in cids}
        all_cands_per_frame = []
        for f_i in range(n_total_frames):
            raw_imgs = {}
            for cid in cids:
                ok, frame_bgr = caps[cid].read()
                if not ok:
                    raise RuntimeError(f"Failed to read frame {f_i} from {cid}.mp4")
                raw_imgs[cid] = frame_bgr
            if f_i in sub_cands:
                all_cands_per_frame.append(sub_cands[f_i])
                continue
            view_cands = {}
            for cid in cids:
                img = cv2.resize(raw_imgs[cid], (1280, 720), interpolation=cv2.INTER_AREA)
                boxes = body.det_model(img)
                cands = []
                for box in boxes:
                    kps, scores = body.pose_model(img, bboxes=[box])
                    bkps, bscores = body.body_pose_model(img, bboxes=[box])
                    kps[:, :17] = bkps
                    scores[:, :17] = bscores
                    cands.append({
                        "box": box.tolist(),
                        "kps": kps[0].tolist(),
                        "scores": scores[0].tolist(),
                    })
                view_cands[cid] = cands
            all_cands_per_frame.append(view_cands)
            if (f_i + 1) % 30 == 0 or f_i == 0:
                print(f"  2D Inference progress: frame {f_i + 1:03d}/{n_total_frames}")
        for cap in caps.values():
            cap.release()
        full_cache_path.write_bytes(pickle.dumps(all_cands_per_frame))
        print(f"  Saved 426-frame 2D detections to {full_cache_path.name}")

    def point_to_ray_dist(pt: np.ndarray, cam_c: np.ndarray, cam_r: np.ndarray) -> float:
        v = pt - cam_c
        t_proj = max(0.0, float(np.dot(v, cam_r)))
        return float(np.linalg.norm(pt - (cam_c + t_proj * cam_r)))

    pos_B = np.array([0.72, 5.15, 1.12])
    pos_A = np.array([-0.85, 4.42, 1.18])
    boxes_B: dict[str, np.ndarray] = {}
    boxes_A: dict[str, np.ndarray] = {}
    frame_B: dict[str, int] = {}
    frame_A: dict[str, int] = {}
    target_history = []

    for f_i, view_cands in enumerate(all_cands_per_frame):
        # 2a. Competitive bipartite per-camera assignment between Track B (Target) and Track A (Interactor)
        cand_assign: dict[str, dict[str, int]] = {c: {} for c in cids}
        for c in cids:
            cands = view_cands[c]
            if not cands:
                continue
            costs_B, costs_A = [], []
            for idx, cand in enumerate(cands):
                ray_d_B = [
                    point_to_ray_dist(pos_B, *get_camera_ray(cameras_cfg[c], cand["kps"][ai]))
                    for ai in anchor_indices if cand["scores"][ai] >= 0.3
                ]
                ray_d_A = [
                    point_to_ray_dist(pos_A, *get_camera_ray(cameras_cfg[c], cand["kps"][ai]))
                    for ai in anchor_indices if cand["scores"][ai] >= 0.3
                ]
                md_B = float(np.mean(ray_d_B)) if len(ray_d_B) >= 2 else 999.0
                md_A = float(np.mean(ray_d_A)) if len(ray_d_A) >= 2 else 999.0
                bx = np.array(cand["box"][:4], dtype=float)
                center = 0.5 * (bx[:2] + bx[2:4])
                jmp_B = float(np.linalg.norm(center - 0.5 * (boxes_B[c][:2] + boxes_B[c][2:4]))) if c in boxes_B else 0.0
                jmp_A = float(np.linalg.norm(center - 0.5 * (boxes_A[c][:2] + boxes_A[c][2:4]))) if c in boxes_A else 0.0
                costs_B.append((md_B, jmp_B, md_B + 0.004 * jmp_B))
                costs_A.append((md_A, jmp_A, md_A + 0.004 * jmp_A))

            best_pair = (None, None)
            best_total = 9999.0
            opts_B = [
                i for i in range(len(cands))
                if costs_B[i][0] < 0.55 and (c not in boxes_B or costs_B[i][1] < 65.0 + 15.0 * min(f_i - frame_B[c] - 1, 15))
            ] + [None]
            opts_A = [
                i for i in range(len(cands))
                if costs_A[i][0] < 0.55 and (c not in boxes_A or costs_A[i][1] < 65.0 + 15.0 * min(f_i - frame_A[c] - 1, 15))
            ] + [None]
            for iB in opts_B:
                for iA in opts_A:
                    if iB is not None and iA is not None and iB == iA:
                        continue
                    cB = costs_B[iB][2] if iB is not None else 0.85
                    cA = costs_A[iA][2] if iA is not None else 0.85
                    if cB + cA < best_total:
                        best_total = cB + cA
                        best_pair = (iB, iA)
            if best_pair[0] is not None:
                cand_assign[c]["B"] = best_pair[0]
                boxes_B[c] = np.array(cands[best_pair[0]]["box"][:4], dtype=float)
                frame_B[c] = f_i
            if best_pair[1] is not None:
                cand_assign[c]["A"] = best_pair[1]
                boxes_A[c] = np.array(cands[best_pair[1]]["box"][:4], dtype=float)
                frame_A[c] = f_i

        # 2b. Update 3D positions pos_B and pos_A from multi-view epipolar intersections
        for label in ("B", "A"):
            mids, weights = [], []
            active_c = [c for c in cids if label in cand_assign[c]]
            for i1 in range(len(active_c)):
                for i2 in range(i1 + 1, len(active_c)):
                    c1, c2 = active_c[i1], active_c[i2]
                    cand1 = view_cands[c1][cand_assign[c1][label]]
                    cand2 = view_cands[c2][cand_assign[c2][label]]
                    for ai in anchor_indices:
                        if cand1["scores"][ai] >= 0.3 and cand2["scores"][ai] >= 0.3:
                            C1, r1 = get_camera_ray(cameras_cfg[c1], cand1["kps"][ai])
                            C2, r2 = get_camera_ray(cameras_cfg[c2], cand2["kps"][ai])
                            d, mid = ray_distance_and_midpoint(C1, r1, C2, r2)
                            if d < 0.35 and mid is not None and mid[2] > -0.25:
                                mids.append(mid)
                                weights.append(1.0 / (d + 0.02))
            if len(mids) >= 2:
                est = np.average(mids, axis=0, weights=np.array(weights))
                if label == "B":
                    pos_B = 0.55 * pos_B + 0.45 * est
                else:
                    pos_A = 0.55 * pos_A + 0.45 * est

        matched_cands = {c: cand_assign[c]["B"] for c in cids if "B" in cand_assign[c]}
        target_history.append({
            "frame_id": f_i,
            "target_pos": pos_B.copy(),
            "matched_cands": matched_cands,
            "all_cands": view_cands,
        })
        if (f_i + 1) % 60 == 0 or f_i == 0 or f_i == n_total_frames - 1:
            print(f"  Frame {f_i:03d}/{n_total_frames}: Locked Target B 3D Pos: {np.round(pos_B, 2)}, Views: {list(matched_cands.keys())}")

    print("=== Step 3: Robust 3D Triangulation & Causal Filtering ===")
    body12_names = [
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]
    all_33_names = names

    states_3d = []
    prev_joints_world = None

    for f_i, entry in enumerate(target_history):
        matched = entry["matched_cands"]
        all_cands = entry["all_cands"]
        joints_3d = {}

        used_cids = [c for c in cids if c in matched]
        if len(used_cids) >= 2:
            sub_cams = [cameras_cfg[c] for c in used_cids]
            geom = CameraGeometry(sub_cams)
            for j_name in all_33_names:
                j_idx = name_to_idx[j_name]
                pts = [all_cands[c][matched[c]]["kps"][j_idx] for c in used_cids]
                scs = [all_cands[c][matched[c]]["scores"][j_idx] for c in used_cids]
                fit = triangulate(sub_cams, pts, scs, threshold_px=15.0, min_angle_deg=2.0, geometry=geom, solver="newton")
                if fit is not None and fit.get("xyz_m") is not None:
                    joints_3d[j_name] = np.array(fit["xyz_m"])
                else:
                    joints_3d[j_name] = None
        else:
            for j_name in all_33_names:
                joints_3d[j_name] = None

        if prev_joints_world is not None:
            for j_name in body12_names:
                if joints_3d.get(j_name) is None or np.isnan(joints_3d[j_name]).any():
                    joints_3d[j_name] = prev_joints_world[j_name].copy()
                else:
                    joints_3d[j_name] = 0.75 * joints_3d[j_name] + 0.25 * prev_joints_world[j_name]
        else:
            # Fallback initialization for frame 0 if any peripheral joint is occluded
            ref_c = entry["target_pos"]
            defaults = {
                "left_shoulder": ref_c + [0.18, 0.0, 0.35], "right_shoulder": ref_c + [-0.18, 0.0, 0.35],
                "left_elbow": ref_c + [0.24, 0.0, 0.05], "right_elbow": ref_c + [-0.24, 0.0, 0.05],
                "left_wrist": ref_c + [0.22, 0.0, -0.20], "right_wrist": ref_c + [-0.22, 0.0, -0.20],
                "left_hip": ref_c + [0.10, 0.0, -0.22], "right_hip": ref_c + [-0.10, 0.0, -0.22],
                "left_knee": ref_c + [0.10, 0.0, -0.60], "right_knee": ref_c + [-0.10, 0.0, -0.60],
                "left_ankle": ref_c + [0.10, 0.0, -0.98], "right_ankle": ref_c + [-0.10, 0.0, -0.98],
            }
            for j_name in body12_names:
                if joints_3d.get(j_name) is None or np.isnan(joints_3d[j_name]).any():
                    joints_3d[j_name] = defaults[j_name].copy()
        prev_joints_world = {k: (v.copy() if v is not None else np.zeros(3)) for k, v in joints_3d.items()}

        stamp_ns = int(round(f_i * (1_000_000_000 / 30.0)))
        st_row = {
            "frame_id": f_i,
            "timestamp_ns": stamp_ns,
            "source_id": "single_person",
            "landmark_schema_id": "human33",
            "calibration_sha256": "mamma_outdoor_calib",
            "perception_signature": "mamma_rtmpose_v1",
            "state_schema_id": "continuous_human_state_v1",
            "joint_names": body12_names,
            "state_timestamp_ns": stamp_ns,
            "hardware_enabled": False,
            "reset": False,
            "joints_world_m": [joints_3d[n].tolist() if n in joints_3d and joints_3d[n] is not None else [0.0, 0.0, 0.0] for n in body12_names],
            "joint_states": [
                {
                    "estimate_status": "updated",
                    "measurement_status": "available",
                    "control_permission": True,
                    "last_observed_timestamp_ns": stamp_ns,
                    "prediction_age_ns": 0,
                    "uncertainty_scale_m": 0.025,
                    "robust_weight": 1.0,
                    "solver_ok": True,
                    "structure_ok": True,
                }
                for _ in body12_names
            ],
            "all_33_joints": {n: (joints_3d[n].tolist() if n in joints_3d and joints_3d[n] is not None else None) for n in all_33_names},
        }
        states_3d.append(st_row)

    print(f"Triangulated 3D states for {len(states_3d)} frames.")

    print("=== Step 4: BRM-C2 Kinematic Retargeting to G1 EDU (29 DoF) ===")
    plant = TwistPlant()
    retarget_cfg = json.loads((ROOT / "config/continuous_prediction.json").read_text(encoding="utf-8"))
    rows_25hz, stream_50hz = build_brm_c2_reference_stream(states_3d, cfg=retarget_cfg, plant=plant, dt=0.02)
    n_ticks = len(stream_50hz["bounded"])
    n_acc = sum(1 for r in rows_25hz if r.get("accepted") and not r.get("short_gap_held"))
    n_held = sum(1 for r in rows_25hz if r.get("short_gap_held"))
    n_rej = sum(1 for r in rows_25hz if not r.get("accepted"))
    reasons = {}
    for r in rows_25hz:
        rs = r.get("reason") or r.get("hold_reason")
        if rs:
            reasons[rs] = reasons.get(rs, 0) + 1
    print(f"Retargeting summary ({n_total_frames} source frames): fresh_accepted={n_acc}, short_gap_held={n_held}, rejected={n_rej}, reasons={reasons}")
    max_track_tick = min(n_ticks, 150 + int(round((n_total_frames - 1) / 30.0 / 0.02)) + 1)
    print(f"Generated 50 Hz BRM-C2 reference stream: {n_ticks} ticks (tracking window: tick 150..{max_track_tick - 1})")

    print("=== Step 5: Genuine Full-Dynamics Closed-Loop MuJoCo Simulation (1000 Hz Physics, PD Torque Control) ===")
    mj = plant.mj
    data = mj.MjData(plant.model)

    def run_closed_loop(controller_name: str):
        sim_plant = Plant()
        policy = FACTORIES[controller_name](sim_plant)
        qpos_trace = []
        tilt_deg_trace = []
        upright_trace = []
        for tick in range(n_ticks):
            ix = window_indices(tick, 0, n_ticks)
            window = [packet(stream_50hz, int(j)) for j in ix]
            sim_plant.data.xfrc_applied[:] = 0.0
            target = policy.step(window, sim_plant)
            rec = sim_plant.advance(target, policy.kp, policy.kd)
            qp = rec["qpos"].copy()
            qpos_trace.append(qp)
            up_vec = Rotation.from_quat(qp[[4, 5, 6, 3]]).apply([0.0, 0.0, 1.0])
            tilt_deg = float(np.degrees(np.arccos(np.clip(up_vec[2], -1.0, 1.0))))
            tilt_deg_trace.append(tilt_deg)
            is_viol = bool(balance_violation(qp[None, :])[0])
            upright_trace.append(not is_viol)
        track_slice = slice(150, max_track_tick)
        z_arr = np.array([q[2] for q in qpos_trace[track_slice]])
        up_arr = np.array(upright_trace[track_slice])
        tilt_arr = np.array(tilt_deg_trace[track_slice])
        first_fall = next((i for i, u in enumerate(up_arr) if not u), None)
        print(
            f"  [{controller_name}] Tracking ticks 150..{track_slice.stop - 1}: "
            f"upright_frac={up_arr.mean()*100:.1f}%, min_z={z_arr.min():.3f}m, "
            f"max_tilt={tilt_arr.max():.1f} deg, first_fall_offset_tick={first_fall}"
        )
        return qpos_trace, tilt_deg_trace, upright_trace

    twist_qpos, twist_tilt, twist_upright = run_closed_loop("twist")
    sonic_qpos, sonic_tilt, sonic_upright = run_closed_loop("sonic")
    sim_qpos = twist_qpos
    sim_tilt = twist_tilt
    sim_upright = twist_upright

    print("=== Step 6: Rendering 4-Stage Synchronous Full Video & Keyframes ===")
    video_path = OUTPUT_DIR / "mamma_4stage_showcase.mp4"
    width, height = 1600, 900
    tile_w, tile_h = 760, 360

    plant.model.vis.global_.offwidth = tile_w
    plant.model.vis.global_.offheight = tile_h
    renderer = mj.Renderer(plant.model, height=tile_h, width=tile_w)
    mcam = mj.MjvCamera()
    mcam.azimuth = 135
    mcam.elevation = -14
    mcam.distance = 2.50

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", "30",
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    keyframe_indices = [30, 125, 260, 390]
    keyframe_names = [
        "keyframe_01_standing.png",
        "keyframe_02_deep_squat.png",
        "keyframe_03_lift_reach.png",
        "keyframe_04_recovered_standing.png",
    ]
    render_caps = {cid: cv2.VideoCapture(str(video_dir / f"{cid}.mp4")) for cid in cids}

    for f_i in range(n_total_frames):
        now_s = f_i / 30.0
        # Align 30 FPS video time now_s with the 50 Hz tracking phase starting at t = 3.0 s (tick 150)
        tick_50 = min(150 + int(round(now_s / 0.02)), len(stream_50hz["bounded"]) - 1)

        cur_rgb_1280 = {}
        for cid in cids:
            ok, bgr_raw = render_caps[cid].read()
            if not ok:
                raise RuntimeError(f"Render read failed at frame {f_i} for {cid}")
            bgr_1280 = cv2.resize(bgr_raw, (1280, 720), interpolation=cv2.INTER_AREA)
            cur_rgb_1280[cid] = cv2.cvtColor(bgr_1280, cv2.COLOR_BGR2RGB)

        canvas = Image.new("RGB", (width, height), "#f4f6f8")
        draw = ImageDraw.Draw(canvas)

        # Header Title
        draw.text((24, 12), "复杂户外多人场景主线全链路四阶段同步演示 — MAMMA 完整视频 (14.20s / 426帧)", font=fonts[22], fill="#17283c")
        draw.text(
            (24, 46),
            f"帧号 f = {f_i:03d}/{n_total_frames} (t = {now_s:5.2f} s / 14.20 s)  |  算法：跨视角极线支持度+因果2D门限 -> 鲁棒三角化 -> BRM-C2重定向 -> MuJoCo 1000Hz力矩闭环",
            font=fonts[15],
            fill="#4a5d70",
        )

        # --- Tile 1: 4-Camera Raw Multi-View Video with Detection & Target Locked ---
        t1 = Image.new("RGB", (tile_w, tile_h), "#101820")
        t1_draw = ImageDraw.Draw(t1)
        sub_w, sub_h = 380, 180
        for ci_idx, cid in enumerate(cids):
            r, c = divmod(ci_idx, 2)
            x0, y0 = c * sub_w, r * sub_h
            rgb_res = cv2.resize(cur_rgb_1280[cid], (sub_w, sub_h), interpolation=cv2.INTER_AREA)

            cand_info = target_history[f_i]["all_cands"][cid]
            locked_idx = target_history[f_i]["matched_cands"].get(cid, None)
            sx, sy = sub_w / 1280.0, sub_h / 720.0
            for b_i, cand in enumerate(cand_info):
                bx = cand["box"]
                p0 = (int(bx[0] * sx), int(bx[1] * sy))
                p1 = (int(bx[2] * sx), int(bx[3] * sy))
                is_target = (b_i == locked_idx)
                col = (46, 204, 113) if is_target else (231, 76, 60)
                thick = 2 if is_target else 1
                cv2.rectangle(rgb_res, p0, p1, col, thick)
                if is_target:
                    cv2.putText(rgb_res, "TARGET LOCK", (p0[0], max(12, p0[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (46, 204, 113), 1)

            t1.paste(Image.fromarray(rgb_res), (x0, y0))
            t1_draw.rectangle([x0 + 4, y0 + 4, x0 + 130, y0 + 22], fill=(16, 24, 32))
            t1_draw.text((x0 + 8, y0 + 6), f"Camera {cid}", font=fonts[13], fill="#ffffff")

        t1_draw.line([(sub_w, 0), (sub_w, tile_h)], fill="#2c3e50", width=2)
        t1_draw.line([(0, sub_h), (tile_w, sub_h)], fill="#2c3e50", width=2)
        canvas.paste(t1, (20, 106))
        draw.text((20, 80), "① 阶段一：户外 4 相机原始视频与跨视角目标锁定 (绿框=锁定受试者 B)", font=fonts[16], fill="#1f4e78")
        draw.text((20, 470), "输入：真实户外草坪/强光阴影/双人推搡倒地与拉起交互，极线+时序门限稳定锁定目标", font=fonts[14], fill="#2c3e50")

        # --- Tile 2: 2D Keypoints & 3D Isometric Skeleton ---
        t2 = Image.new("RGB", (tile_w, tile_h), "#16202a")
        t2_draw = ImageDraw.Draw(t2)
        for r_i, cid in enumerate(("A001", "B001")):
            y_off = r_i * 180
            sub_crop = cv2.resize(cur_rgb_1280[cid], (380, 180), interpolation=cv2.INTER_AREA)
            locked_idx = target_history[f_i]["matched_cands"].get(cid, None)
            if locked_idx is not None:
                cand = target_history[f_i]["all_cands"][cid][locked_idx]
                kps = cand["kps"]
                scores = cand["scores"]
                sx, sy = 380.0 / 1280.0, 180.0 / 720.0
                pts_2d = {
                    name_to_idx[n]: (int(kps[name_to_idx[n]][0] * sx), int(kps[name_to_idx[n]][1] * sy))
                    for n in all_33_names if scores[name_to_idx[n]] >= 0.25
                }
                for n1, n2 in SKELETON_EDGES:
                    i1, i2 = name_to_idx.get(n1), name_to_idx.get(n2)
                    if i1 in pts_2d and i2 in pts_2d:
                        col = (46, 204, 113) if ("left" in n1 or "left" in n2) else (52, 152, 219)
                        cv2.line(sub_crop, pts_2d[i1], pts_2d[i2], col, 2, cv2.LINE_AA)
                for pt in pts_2d.values():
                    cv2.circle(sub_crop, pt, 2, (241, 196, 15), -1, cv2.LINE_AA)
            t2.paste(Image.fromarray(sub_crop), (0, y_off))
            t2_draw.rectangle([6, y_off + 6, 210, y_off + 26], fill=(16, 24, 32))
            t2_draw.text((10, y_off + 8), f"Cam {cid}: 2D RTMPose 关键点", font=fonts[13], fill="#ffffff")

        # Right half (380x360): 3D Isometric projection
        t2_draw.rectangle([380, 0, tile_w, tile_h], fill="#f8f9fa")
        st_map = {n: np.array(p) for n, p in states_3d[f_i]["all_33_joints"].items() if p is not None}
        cur_pos = target_history[f_i]["target_pos"]

        az = math.radians(45.0)
        el = math.radians(20.0)
        ca, sa = math.cos(az), math.sin(az)
        ce, se = math.cos(el), math.sin(el)

        def proj_3d(p_world: np.ndarray) -> tuple[int, int]:
            dp = p_world - cur_pos
            x, y, z = float(dp[0]), float(dp[1]), float(dp[2])
            u = -sa * x + ca * y
            v = -se * ca * x - se * sa * y + ce * z
            sx = int(round(570 + u * 150.0))
            sy = int(round(190 - v * 150.0))
            return max(384, min(756, sx)), max(28, min(328, sy))

        z_fl = cur_pos[2] - 0.90
        for gx in np.linspace(-0.8, 0.8, 5):
            p0 = proj_3d(np.array([cur_pos[0] + gx, cur_pos[1] - 0.8, z_fl]))
            p1 = proj_3d(np.array([cur_pos[0] + gx, cur_pos[1] + 0.8, z_fl]))
            t2_draw.line([p0, p1], fill="#dde3ea", width=1)
        for gy in np.linspace(-0.8, 0.8, 5):
            p0 = proj_3d(np.array([cur_pos[0] - 0.8, cur_pos[1] + gy, z_fl]))
            p1 = proj_3d(np.array([cur_pos[0] + 0.8, cur_pos[1] + gy, z_fl]))
            t2_draw.line([p0, p1], fill="#dde3ea", width=1)

        for n1, n2 in SKELETON_EDGES:
            if n1 in st_map and n2 in st_map:
                p1, p2 = proj_3d(st_map[n1]), proj_3d(st_map[n2])
                col = "#1b9e77" if ("left" in n1 or "left" in n2) else "#2b83ba"
                t2_draw.line([p1, p2], fill=col, width=3)
        for pw in st_map.values():
            px, py = proj_3d(pw)
            t2_draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill="#d95f02", outline="#ffffff")

        t2_draw.text((392, 10), "多视角三角化 + 因果滤波 3D 骨架 (世界系 m)", font=fonts[14], fill="#17283c")
        t2_draw.text((392, 332), f"锁定目标 3D 坐标: [{cur_pos[0]:.2f}, {cur_pos[1]:.2f}, {cur_pos[2]:.2f}] m", font=fonts[13], fill="#4a5d70")
        t2_draw.line([(380, 0), (380, tile_h)], fill="#2c3e50", width=2)
        canvas.paste(t2, (805, 106))
        draw.text((805, 80), "② 阶段二：多视角 2D 关键点检测与 3D 空间骨架重建", font=fonts[16], fill="#2e75b6")
        draw.text((805, 470), "左：Cam A / Cam B 二维姿态叠加  |  右：世界坐标系下三维骨架等轴测投影", font=fonts[14], fill="#2c3e50")

        # --- Tile 3: Robot Kinematic Retargeting (BRM-C2) ---
        q_kin = stream_50hz["bounded"][tick_50].copy()
        cur_stage = str(stream_50hz["stage"][tick_50])
        src_row = rows_25hz[f_i]
        is_held = bool(src_row.get("short_gap_held", False))
        is_acc = bool(src_row.get("accepted", False))
        ref_status_str = "实时跟踪(Tracking)" if (is_acc and not is_held) else ("短时保持(Held)" if is_held else "超界保护回退(Withdrawn)")
        data.qpos[:] = q_kin
        mj.mj_kinematics(plant.model, data)
        mcam.lookat[:] = [q_kin[0], q_kin[1], 0.65]
        renderer.update_scene(data, camera=mcam)
        t3 = Image.fromarray(renderer.render())
        t3_draw = ImageDraw.Draw(t3)
        t3_draw.rectangle([10, 10, 460, 36], fill=(16, 24, 32))
        t3_draw.text((16, 13), f"G1 29-DoF 运动学参考 (tick={tick_50}, 状态: {ref_status_str})", font=fonts[13], fill="#ffffff")
        canvas.paste(t3, (20, 521))
        draw.text((20, 495), "③ 阶段三：宇树 G1 EDU 机器人运动学重定向 (BRM-C2 奇异壁垒)", font=fonts[16], fill="#1f4e78")
        draw.text((20, 880), f"参考骨盆高度: {q_kin[2]:.3f} m (下限钳位 0.48m, 直立基座先验)  |  阶段: {cur_stage}", font=fonts[14], fill="#2c3e50")

        # --- Tile 4: Robot Closed-Loop Dynamic Physical Tracking (MuJoCo) ---
        q_dyn = sim_qpos[tick_50].copy()
        cur_tilt = sim_tilt[tick_50]
        cur_up = sim_upright[tick_50]
        bal_str = "平衡正常 (Upright)" if cur_up else "失稳倾覆 (Balance Violated!)"
        bal_col = (46, 204, 113) if cur_up else (231, 76, 60)
        data.qpos[:] = q_dyn
        mj.mj_kinematics(plant.model, data)
        mcam.lookat[:] = [q_dyn[0], q_dyn[1], 0.65]
        renderer.update_scene(data, camera=mcam)
        t4 = Image.fromarray(renderer.render())
        t4_draw = ImageDraw.Draw(t4)
        t4_draw.rectangle([10, 10, 520, 36], fill=(16, 24, 32))
        t4_draw.text((16, 13), f"MuJoCo 1000Hz 真实 PD 力矩闭环 (TWIST2) — {bal_str}", font=fonts[13], fill=bal_col)
        canvas.paste(t4, (805, 521))
        draw.text((805, 495), "④ 阶段四：机器人真实全动力学力矩闭环跟踪 (无状态重置, 1000 Hz 物理)", font=fonts[16], fill="#2e75b6")
        min_z_so_far = min([q[2] for q in sim_qpos[150:tick_50+1]])
        draw.text((805, 880), f"实际骨盆高度: {q_dyn[2]:.3f} m (累计最低: {min_z_so_far:.3f} m)  |  躯干倾角: {cur_tilt:.1f}°", font=fonts[14], fill="#2c3e50")

        # Save keyframe images
        if f_i in keyframe_indices:
            idx = keyframe_indices.index(f_i)
            canvas.save(FIG_DIR / keyframe_names[idx])
            print(f"Saved keyframe: {keyframe_names[idx]}")

        proc.stdin.write(canvas.tobytes())

    for cap in render_caps.values():
        cap.release()
    proc.stdin.close()
    proc.wait()
    print(f"\nShowcase video successfully rendered to: {video_path}")


if __name__ == "__main__":
    main()
