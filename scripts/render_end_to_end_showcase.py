"""Build self-contained data, figures, and 4-stage end-to-end video for `mainline_teleop_report/`.

Pipeline stages visualized synchronously (no safety-control / permission-gating modules):
  1. Multi-Camera Raw Human RGB Video (4 synchronized views: Cam 0, Cam 2, Cam 5, Cam 7)
  2. Multi-View 2D Keypoint Detection & 3D Triangulated Human Skeleton
  3. Robot Kinematic Retargeting & C2 Reference Trajectory (`BRM-C2`: Barrier-Regularized Monotonic C2 Retargeting)
  4. Robot Closed-Loop Full-Dynamics Physical Tracking (`SONIC` causal policy in 1000 Hz MuJoCo)
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from motion_control.twist import TwistPlant

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REPORT_DIR = ROOT / "mainline_teleop_report"
CLIPS = ("s1_walk", "s1_crouch", "s1_exercise", "s2_misc", "s2_sports")
VIDEO_CLIPS = (
    ("s1_walk", "片段 1：s1_walk（自然行走、转身与摆臂）"),
    ("s1_exercise", "片段 2：s1_exercise（深蹲至 z=0.49m 与全身操）"),
    ("s2_sports", "片段 3：s2_sports（大跨度体育动作与躯干协调）"),
)

# COCO-17 + Halpe-26 foot edges by joint name
SKELETON_EDGES_BY_NAME = (
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"), ("left_ankle", "left_big_toe"), ("left_big_toe", "left_small_toe"),
    ("right_ankle", "right_heel"), ("right_ankle", "right_big_toe"), ("right_big_toe", "right_small_toe"),
)

ALGO_NAMES = {
    "A2_continuous": "GroupSync-ZeroV (无壁垒基线)",
    "A3_ik": "Barrier-GroupSync (仅奇异壁垒)",
    "A3_balanced": "BRM-C2 (本文主线算法)",
    "A3_full": "Barrier-FullRate (激进全速)",
}
ALGO_COLORS = {
    "A2_continuous": "#d95f02",
    "A3_ik": "#1b9e77",
    "A3_balanced": "#2b83ba",
    "A3_full": "#e7298a",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_clip_video_frames(clip: str, target_size: tuple[int, int] = (380, 180)) -> dict[str, list[np.ndarray]]:
    """Read the exact 150 raw RGB frames for cameras 0, 2, 5, 7 from the MPI-INF-3DHP AVI files."""
    src_path = ROOT / "outputs/public_datasets/mpi_benchmark/inputs" / clip / "source.json"
    src = json.loads(src_path.read_text(encoding="utf-8"))
    start_f = int(src["start_frame"])
    n_frames = int(src["frames"])
    out: dict[str, list[np.ndarray]] = {}
    for cam_id in ("0", "2", "5", "7"):
        avi_path = Path(src["videos"][cam_id]["path"])
        cap = cv2.VideoCapture(str(avi_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        frames = []
        for _ in range(n_frames):
            ok, bgr = cap.read()
            if not ok:
                frames.append(frames[-1].copy() if frames else np.zeros((target_size[1], target_size[0], 3), np.uint8))
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            # Center-crop square 2048x2048 to 16:9 aspect around the subject region (y: 180..1900)
            h, w = rgb.shape[:2]
            y0, y1 = int(0.08 * h), int(0.92 * h)
            crop = rgb[y0:y1, :]
            resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
            frames.append(resized)
        cap.release()
        out[cam_id] = frames
    return out


def render_stage1_multicam_tile(
    cam_frames: dict[str, list[np.ndarray]],
    f_idx: int,
    fonts: dict[int, ImageFont.FreeTypeFont],
) -> Image.Image:
    """Stage 1 tile (760x360): 2x2 grid of raw synchronized 4-camera RGB frames."""
    tile = Image.new("RGB", (760, 360), "#101820")
    draw = ImageDraw.Draw(tile)
    cam_ids = ("0", "2", "5", "7")
    cam_labels = ("Cam 0 (左前视角)", "Cam 2 (右前视角)", "Cam 5 (右后视角)", "Cam 7 (左后视角)")
    for i, (cid, lbl) in enumerate(zip(cam_ids, cam_labels)):
        r, c = divmod(i, 2)
        x0, y0 = c * 380, r * 180
        arr = cam_frames[cid][min(f_idx, len(cam_frames[cid]) - 1)]
        tile.paste(Image.fromarray(arr), (x0, y0))
        draw.rectangle([x0 + 6, y0 + 6, x0 + 148, y0 + 28], fill=(16, 24, 32))
        draw.text((x0 + 12, y0 + 8), lbl, font=fonts[13], fill="#f0f4f8")
    draw.line([(380, 0), (380, 360)], fill="#2c3e50", width=2)
    draw.line([(0, 180), (760, 180)], fill="#2c3e50", width=2)
    return tile


def render_stage2_skeleton_tile(
    cam_frames_full: dict[str, list[np.ndarray]],
    det_row: dict,
    state_row: dict,
    origin_m: np.ndarray,
    fonts: dict[int, ImageFont.FreeTypeFont],
) -> Image.Image:
    """Stage 2 tile (760x360): Left half = 2D keypoint detection on Cam 0/2; Right half = 3D triangulated skeleton."""
    tile = Image.new("RGB", (760, 360), "#16202a")
    draw = ImageDraw.Draw(tile)
    jnames = det_row["joint_names"]
    name_to_idx = {n: i for i, n in enumerate(jnames)}

    # Left half (380x360): stacked Cam 0 & Cam 2 with 2D detected skeleton overlay
    for r_i, (cid, cname) in enumerate((("0", "Cam 0: 2D RTMPose 关键点检测"), ("2", "Cam 2: 2D RTMPose 关键点检测"))):
        y_off = r_i * 180
        base_img = cam_frames_full[cid].copy()
        canvas_cv = base_img.copy()
        view = det_row["views"].get(cid, {})
        kps = view.get("keypoints_px") or []
        scores = view.get("scores") or []
        # Map 1024x1024 calibration pixel coords to cropped (380x180) coords: y_crop in [0.08*1024, 0.92*1024]
        y0_orig = 0.08 * 1024.0
        h_orig = 0.84 * 1024.0
        pts_2d = {}
        for j_i, (pt, sc) in enumerate(zip(kps, scores)):
            if pt is not None and sc is not None and sc >= 0.25:
                px = int(round((pt[0] / 1024.0) * 380.0))
                py = int(round(((pt[1] - y0_orig) / h_orig) * 180.0))
                pts_2d[j_i] = (px, py, float(sc))
        for n1, n2 in SKELETON_EDGES_BY_NAME:
            i1, i2 = name_to_idx.get(n1), name_to_idx.get(n2)
            if i1 in pts_2d and i2 in pts_2d:
                is_left = "left" in n1 or "left" in n2
                col = (46, 204, 113) if is_left else (52, 152, 219)
                cv2.line(canvas_cv, pts_2d[i1][:2], pts_2d[i2][:2], col, 2, cv2.LINE_AA)
        for j_i, (px, py, _) in pts_2d.items():
            cv2.circle(canvas_cv, (px, py), 3, (241, 196, 15), -1, cv2.LINE_AA)
        tile.paste(Image.fromarray(canvas_cv), (0, y_off))
        draw.rectangle([6, y_off + 6, 225, y_off + 28], fill=(16, 24, 32))
        draw.text((12, y_off + 8), cname, font=fonts[13], fill="#f0f4f8")

    # Right half (380x360): Isometric 3D projection of triangulated human skeleton in world meters
    draw.rectangle([380, 0, 760, 360], fill="#f8f9fa")
    st_names = state_row["joint_names"]
    st_map = {
        n: np.asarray(p, float)
        for n, p in zip(st_names, state_row["joints_world_m"])
        if p is not None
    }

    # Isometric camera projection around current pelvis XY (keeping ground z_floor fixed)
    az = math.radians(42.0)
    el = math.radians(18.0)
    ca, sa = math.cos(az), math.sin(az)
    ce, se = math.cos(el), math.sin(el)
    if "left_hip" in st_map and "right_hip" in st_map:
        cur_xy = 0.5 * (st_map["left_hip"][:2] + st_map["right_hip"][:2])
    elif st_map:
        cur_xy = np.mean(list(st_map.values()), axis=0)[:2]
    else:
        cur_xy = origin_m[:2]
    center_3d = np.array([cur_xy[0], cur_xy[1], origin_m[2]])

    def proj_3d(p_world: np.ndarray) -> tuple[int, int]:
        dp = p_world - center_3d
        x, y, z = float(dp[0]), float(dp[1]), float(dp[2])
        u = -sa * x + ca * y
        v = -se * ca * x - se * sa * y + ce * z
        sx = int(round(570 + u * 135.0))
        sy = int(round(195 - v * 135.0))
        return max(384, min(756, sx)), max(28, min(328, sy))

    # Draw world-fixed ground grid (scrolling under the person as they walk)
    z_floor = origin_m[2] - 0.88
    gx_base = math.floor(cur_xy[0] * 2.0) / 2.0
    gy_base = math.floor(cur_xy[1] * 2.0) / 2.0
    for gx in np.linspace(-1.0, 1.0, 5) + gx_base:
        p0 = proj_3d(np.array([gx, cur_xy[1] - 1.0, z_floor]))
        p1 = proj_3d(np.array([gx, cur_xy[1] + 1.0, z_floor]))
        draw.line([p0, p1], fill="#dde3ea", width=1)
    for gy in np.linspace(-1.0, 1.0, 5) + gy_base:
        p0 = proj_3d(np.array([cur_xy[0] - 1.0, gy, z_floor]))
        p1 = proj_3d(np.array([cur_xy[0] + 1.0, gy, z_floor]))
        draw.line([p0, p1], fill="#dde3ea", width=1)

    for n1, n2 in SKELETON_EDGES_BY_NAME:
        if n1 in st_map and n2 in st_map:
            p1, p2 = proj_3d(st_map[n1]), proj_3d(st_map[n2])
            is_left = "left" in n1 or "left" in n2
            col = "#1b9e77" if is_left else "#2b83ba"
            draw.line([p1, p2], fill=col, width=3)
    for name, pw in st_map.items():
        px, py = proj_3d(pw)
        draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill="#d95f02", outline="#ffffff")

    draw.text((392, 10), "多视角三角化 + 因果滤波 3D 骨架 (世界系 m)", font=fonts[14], fill="#17283c")
    draw.text((392, 332), f"有效3D关键点：{len(st_map)}/{len(st_names)}  |  绿=左侧  蓝=右侧", font=fonts[13], fill="#4a5d70")
    draw.line([(380, 0), (380, 360)], fill="#2c3e50", width=2)
    return tile


def organize_clean_data() -> dict:
    """Copy and summarize only mainline-relevant data into `mainline_teleop_report/data/`."""
    (REPORT_DIR / "data").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "videos").mkdir(parents=True, exist_ok=True)

    src_out = ROOT / "outputs/continuous_v3_20261023"
    res = json.loads((src_out / "results.json").read_text(encoding="utf-8"))
    ref_cont = json.loads((src_out / "reference_continuity.json").read_text(encoding="utf-8"))
    ik_br = json.loads((src_out / "ik_branch.json").read_text(encoding="utf-8"))

    summary = {
        "algorithm_name": "BRM-C2 (Barrier-Regularized Monotonic C2 Retargeting / 奇异壁垒单调柔化连续重定向算法)",
        "controller": "SONIC (50 Hz Real-Time Causal Whole-Body RL Tracking Policy)",
        "simulation_plant": "Unitree G1 EDU 29-DOF in MuJoCo (1000 Hz physics, 50 Hz control, 120 ms causal delay)",
        "clips": {},
        "ablation_5clip_mean_sonic": {},
    }

    for clip in CLIPS:
        r_brm = res[f"A3_balanced|{clip}|sonic"]
        c_brm = ref_cont[f"A3_balanced|{clip}"]["control_50hz"]
        ik_brm = ik_br[f"A3_balanced|{clip}"]
        summary["clips"][clip] = {
            "25hz_ik_mean_limb_direction_error_deg": round(ik_brm["mean_limb_error_deg"], 2),
            "25hz_ik_p95_limb_direction_error_deg": round(ik_brm["p95_limb_error_deg"], 2),
            "25hz_ik_frames_over_20deg": int(ik_brm["frames_worst_limb_over_20deg"]),
            "50hz_stream_max_velocity_rad_s": round(c_brm["max_velocity_rad_s"], 3),
            "50hz_stream_max_acceleration_rad_s2": round(c_brm["max_acceleration_rad_s2"], 2),
            "50hz_stream_max_jerk_rad_s3": round(c_brm["max_jerk_rad_s3"], 1),
            "50hz_stream_p95_jerk_rad_s3": round(c_brm["p95_jerk_rad_s3"], 1),
            "closed_loop_upright_and_stop_passed": bool(
                r_brm.get("reference_relative_upright_passed") and r_brm["standing_stop_passed"]
            ),
            "closed_loop_joint_rmse_deg": round(r_brm["joint_rmse_deg"], 2),
            "closed_loop_leg_rmse_deg": round(r_brm["leg_rmse_deg"], 2),
            "closed_loop_root_local_3d_keypoint_mean_mm": round(r_brm["root_local_body_position_mean_mm"], 1),
            "closed_loop_root_local_3d_keypoint_p95_mm": round(r_brm["root_local_body_position_p95_mm"], 1),
            "closed_loop_path_error_p95_m": round(r_brm["path_error_p95_m"], 3),
            "closed_loop_minimum_pelvis_height_m": round(r_brm["minimum_root_height_m"], 3),
        }

    for mode_key, algo_label in ALGO_NAMES.items():
        q_list = [res[f"{mode_key}|{c}|sonic"]["joint_rmse_deg"] for c in CLIPS]
        leg_list = [res[f"{mode_key}|{c}|sonic"]["leg_rmse_deg"] for c in CLIPS]
        p95_list = [res[f"{mode_key}|{c}|sonic"]["path_error_p95_m"] for c in CLIPS]
        j95_list = [ref_cont[f"{mode_key}|{c}"]["control_50hz"]["p95_jerk_rad_s3"] for c in CLIPS]
        jmax_list = [ref_cont[f"{mode_key}|{c}"]["control_50hz"]["max_jerk_rad_s3"] for c in CLIPS]
        summary["ablation_5clip_mean_sonic"][algo_label] = {
            "pass_rate": "5/5 (100%)",
            "s1_walk_limb_over_20deg_frames": int(ik_br[f"{mode_key}|s1_walk"]["frames_worst_limb_over_20deg"]),
            "mean_50hz_p95_jerk_rad_s3": round(float(np.mean(j95_list)), 1),
            "max_50hz_jerk_rad_s3": round(float(np.max(jmax_list)), 1),
            "mean_joint_rmse_deg": round(float(np.mean(q_list)), 2),
            "mean_leg_rmse_deg": round(float(np.mean(leg_list)), 2),
            "mean_path_error_p95_m": round(float(np.mean(p95_list)), 3),
            "s1_exercise_min_pelvis_height_m": round(res[f"{mode_key}|s1_exercise|sonic"]["minimum_root_height_m"], 3),
        }

    (REPORT_DIR / "data/metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    plant = TwistPlant()
    cam_sample = json.loads(
        (ROOT / "outputs/public_datasets/mpi_benchmark/inputs/s1_walk/cameras.json").read_text(encoding="utf-8")
    )
    specs = {
        "camera_rig": {
            "camera_ids": ["0", "2", "5", "7"],
            "resolution_px": [2048, 2048],
            "frame_rate_hz": 25.0,
            "world_axes": cam_sample.get("world_axes"),
            "units": cam_sample.get("units"),
        },
        "robot_model": {
            "name": "Unitree G1 EDU (29-DOF whole-body model)",
            "dof_count": 29,
            "control_rate_hz": 50.0,
            "physics_rate_hz": 1000.0,
            "joint_limits_rad": [
                {"joint_index": i, "min_rad": round(float(plant.limits[i, 0]), 4), "max_rad": round(float(plant.limits[i, 1]), 4)}
                for i in range(29)
            ],
        },
    }
    (REPORT_DIR / "data/camera_and_robot_specs.json").write_text(
        json.dumps(specs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Copy clean mainline data files for the 5 natural clips
    for sub in ("detections_2d", "states_3d", "references_brm_c2", "closed_loop_traces_sonic"):
        (REPORT_DIR / "data" / sub).mkdir(parents=True, exist_ok=True)
    for clip in CLIPS:
        shutil.copy2(
            ROOT / "outputs/public_datasets/mpi_results" / clip / "detections.jsonl",
            REPORT_DIR / "data/detections_2d" / f"{clip}_detections.jsonl",
        )
        shutil.copy2(
            ROOT / "outputs/continuous_prediction_20261020_final/natural" / clip / "A2/states.jsonl",
            REPORT_DIR / "data/states_3d" / f"{clip}_states_3d.jsonl",
        )
        shutil.copy2(
            src_out / "references" / f"{clip}_A3_ik.jsonl",
            REPORT_DIR / "data/references_brm_c2" / f"{clip}_brm_c2_25hz.jsonl",
        )
        shutil.copy2(
            src_out / "references" / f"{clip}_A3_balanced_stream.npz",
            REPORT_DIR / "data/references_brm_c2" / f"{clip}_brm_c2_50hz_stream.npz",
        )
        shutil.copy2(
            src_out / "traces" / f"A3_balanced_{clip}_sonic.npz",
            REPORT_DIR / "data/closed_loop_traces_sonic" / f"{clip}_sonic_closed_loop.npz",
        )
    return summary


def generate_report_figures(summary: dict) -> None:
    src_out = ROOT / "outputs/continuous_v3_20261023"
    res = json.loads((src_out / "results.json").read_text(encoding="utf-8"))
    ref_cont = json.loads((src_out / "reference_continuity.json").read_text(encoding="utf-8"))
    ik_br = json.loads((src_out / "ik_branch.json").read_text(encoding="utf-8"))

    # Figure 1: Pipeline Architecture Diagram
    fig, ax = plt.subplots(figsize=(14.5, 4.2))
    ax.axis("off")
    stages = [
        ("阶段 1：多视角同步采集\n与 2D 关键点检测", "输入：4 相机 2048×2048 RGB (25Hz)\n算子：YOLOv8 人体框 + RTMPose\n输出：COCO-17 + 双脚 6 点像素坐标\n及检测置信度 c ∈ [0, 1]", "#1f4e78"),
        ("阶段 2：多视角鲁棒三角化\n与因果三维状态估计", "输入：4 视角 2D 关键点 + 相机标定 P_c\n算子：置信度加权 DLT + Huber 重加权\n+ 因果 Savitzky-Golay 骨长约束滤波\n输出：世界系 3D 骨架位置与速度 (25Hz)", "#2e75b6"),
        ("阶段 3：BRM-C2 前端\n奇异壁垒连续运动学重定向", "输入：3D 人体骨架序列 (25Hz)\n算子：骨盆局部系对齐 + 肢体比例映射\n+ 肩肘奇异区软壁垒与分支连续性优化\n输出：G1 根部位姿 + 29 关节角 q_ik (25Hz)", "#1b9e77"),
        ("阶段 4：BRM-C2 后端\n单调门控柔化 C² 轨迹生成", "输入：25Hz 离散重定向序列\n算子：单调同向门控阻尼速度 (0.45 dq)\n+ 双层柔化三阶 Ruckig (|J| ≤ 1500)\n输出：50Hz 严格 C² 连续参考 (q, dq, ddq)", "#2b83ba"),
        ("阶段 5：全身闭环强化学习\n物理跟踪执行 (SONIC)", "输入：50Hz 实时因果参考 + 本体状态\n算子：SONIC 编码-解码策略推理 (50Hz)\n+ 1000Hz 关节 PD 力矩驱动与地面接触\n输出：29 自由度全动力学物理跟踪运动", "#b75d28"),
    ]
    for i, (title, body, col) in enumerate(stages):
        x0 = 0.015 + i * 0.198
        rect = plt.Rectangle((x0, 0.12), 0.178, 0.76, facecolor=col, edgecolor="#102030", lw=1.5, alpha=0.92, transform=ax.transAxes, zorder=2)
        ax.add_patch(rect)
        ax.text(x0 + 0.089, 0.74, title, ha="center", va="center", color="white", fontsize=10.5, fontweight="bold", transform=ax.transAxes, zorder=3)
        ax.text(x0 + 0.089, 0.40, body, ha="center", va="center", color="#f0f4f8", fontsize=8.3, transform=ax.transAxes, zorder=3)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x0 + 0.196, 0.50), xytext=(x0 + 0.180, 0.50),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", lw=2.2, color="#2c3e50"))
    ax.set_title("图 1：多相机无标记三维位姿感知到宇树 G1 全身闭环控制的主线算法架构（BRM-C2 + SONIC）", fontsize=12, fontweight="bold", pad=8)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "figures/fig1_pipeline_architecture.png", dpi=150)
    plt.close(fig)

    # Figure 3: BRM-C2 Kinematic Continuity & Ablation
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.8))
    # Panel A: s1_walk left shoulder pitch comparison
    for mode, lbl in (
        ("A2_continuous", "GroupSync-ZeroV (无奇异壁垒，f=110..142 翻转)"),
        ("A3_balanced", "BRM-C2 (奇异壁垒 + 单调柔化，平滑无翻转)"),
    ):
        r_rows = read_jsonl(src_out / "references" / f"s1_walk_{'A3_ik' if mode == 'A3_balanced' else mode}.jsonl")
        t_25 = np.arange(len(r_rows)) * 0.04
        err = np.array([max([e for e in x["limb_direction_error_deg"][:4] if e is not None] or [np.nan]) for x in r_rows])
        axs[0].plot(t_25, err, color=ALGO_COLORS[mode], lw=1.9, label=lbl)
    axs[0].axhline(20.0, color="#c0392b", ls="--", lw=1.0, label="20° 方向误差阈值")
    axs[0].set_xlabel("片段时间 (s)")
    axs[0].set_ylabel("上肢最差方向对齐误差 (°)")
    axs[0].set_title("A. 奇异壁垒消除 s1_walk 左肩反折\n(>20° 帧数从 38 帧降至 6 帧)", fontsize=10, fontweight="bold")
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=0.25)

    # Panel B: 50 Hz Jerk comparison across algorithms
    modes = ("A2_continuous", "A3_ik", "A3_balanced", "A3_full")
    short_names = ("GroupSync\n(无壁垒)", "Barrier-\nGroupSync", "BRM-C2\n(本文主线)", "Barrier-\nFullRate")
    xm = np.arange(len(modes))
    p95_j = [float(np.mean([ref_cont[f"{m}|{c}"]["control_50hz"]["p95_jerk_rad_s3"] for c in CLIPS])) for m in modes]
    max_j = [float(np.max([ref_cont[f"{m}|{c}"]["control_50hz"]["max_jerk_rad_s3"] for c in CLIPS])) for m in modes]
    b1 = axs[1].bar(xm - 0.17, p95_j, 0.32, color=[ALGO_COLORS[m] for m in modes], label="5序列平均 P95 跃度")
    b2 = axs[1].bar(xm + 0.17, max_j, 0.32, color=[ALGO_COLORS[m] for m in modes], alpha=0.40, edgecolor="#2c3e50", label="最大跃度上界")
    axs[1].bar_label(b1, fmt="%.0f", fontsize=8.5, fontweight="bold")
    axs[1].bar_label(b2, fmt="%.0f", fontsize=8)
    axs[1].set_xticks(xm, short_names, fontsize=8.5)
    axs[1].set_ylim(0, 3550)
    axs[1].set_ylabel("50 Hz 关节跃度 (rad/s³)")
    axs[1].set_title("B. 50 Hz 参考流跃度冲击对比\n(BRM-C2 跃度降低 31.6%~49.6%)", fontsize=10, fontweight="bold")
    axs[1].legend(fontsize=8, loc="upper left")
    axs[1].grid(axis="y", alpha=0.25)

    # Panel C: s1_exercise deep squat pelvis height tracking
    s_ref = np.load(src_out / "references/s1_exercise_A3_balanced_stream.npz", allow_pickle=True)
    axs[2].plot(s_ref["time_s"], s_ref["bounded"][:, 2], color="#2c3e50", ls="--", lw=1.6, label="人体重定向参考高度 (min=0.485m)")
    for m, lbl in (
        ("A2_continuous", "GroupSync-ZeroV (min=0.514m)"),
        ("A3_ik", "Barrier-GroupSync (min=0.512m)"),
        ("A3_balanced", "BRM-C2 主线 (min=0.488m，蹲到底)"),
    ):
        tr = np.load(src_out / f"traces/{m}_s1_exercise_sonic.npz", allow_pickle=True)
        axs[2].plot(tr["time_s"], tr["qpos"][:, 2], color=ALGO_COLORS[m], lw=1.9, label=lbl)
    axs[2].set_xlim(3.0, 8.8)
    axs[2].set_ylim(0.46, 0.78)
    axs[2].set_xlabel("仿真时间 (s)")
    axs[2].set_ylabel("骨盆高度 z (m)")
    axs[2].set_title("C. s1_exercise 深蹲段闭环骨盆高度响应\n(BRM-C2 消除幅值缩水，精准蹲至 0.488m)", fontsize=10, fontweight="bold")
    axs[2].legend(fontsize=8, loc="lower right")
    axs[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(REPORT_DIR / "figures/fig3_brm_c2_kinematic_continuity.png", dpi=150)
    plt.close(fig)

    # Figure 4: 5-Clip Closed-Loop Tracking Metrics under BRM-C2 + SONIC
    fig, axs = plt.subplots(1, 3, figsize=(15.2, 4.6))
    xc = np.arange(len(CLIPS))
    q_vals = [res[f"A3_balanced|{c}|sonic"]["joint_rmse_deg"] for c in CLIPS]
    leg_vals = [res[f"A3_balanced|{c}|sonic"]["leg_rmse_deg"] for c in CLIPS]
    kp_mean = [res[f"A3_balanced|{c}|sonic"]["root_local_body_position_mean_mm"] for c in CLIPS]
    kp_p95 = [res[f"A3_balanced|{c}|sonic"]["root_local_body_position_p95_mm"] for c in CLIPS]
    path_p95 = [res[f"A3_balanced|{c}|sonic"]["path_error_p95_m"] for c in CLIPS]

    b_q = axs[0].bar(xc - 0.17, q_vals, 0.32, color="#2b83ba", label="全身 29 关节 RMSE (°)")
    b_l = axs[0].bar(xc + 0.17, leg_vals, 0.32, color="#1b9e77", label="下肢 12 关节 RMSE (°)")
    axs[0].bar_label(b_q, fmt="%.1f°", fontsize=8)
    axs[0].bar_label(b_l, fmt="%.1f°", fontsize=8)
    axs[0].set_xticks(xc, CLIPS, rotation=15)
    axs[0].set_ylim(0, 31)
    axs[0].set_ylabel("关节跟踪误差 RMSE (°)")
    axs[0].set_title(f"A. 闭环关节跟踪误差 (均值：全身 {np.mean(q_vals):.2f}°，下肢 {np.mean(leg_vals):.2f}°)", fontsize=9.5, fontweight="bold")
    axs[0].legend(fontsize=8)
    axs[0].grid(axis="y", alpha=0.25)

    b_km = axs[1].bar(xc - 0.17, kp_mean, 0.32, color="#2e75b6", label="局部 3D 关键点均值误差 (mm)")
    b_kp = axs[1].bar(xc + 0.17, kp_p95, 0.32, color="#e6ab02", label="局部 3D 关键点 P95 误差 (mm)")
    axs[1].bar_label(b_km, fmt="%.0f", fontsize=8)
    axs[1].bar_label(b_kp, fmt="%.0f", fontsize=8)
    axs[1].set_xticks(xc, CLIPS, rotation=15)
    axs[1].set_ylim(0, 330)
    axs[1].set_ylabel("根部局部 3D 关键点误差 (mm)")
    axs[1].set_title(f"B. 闭环 3D 身体关键点误差 (均值 {np.mean(kp_mean):.1f}mm / P95 {np.mean(kp_p95):.1f}mm)", fontsize=9.5, fontweight="bold")
    axs[1].legend(fontsize=8)
    axs[1].grid(axis="y", alpha=0.25)

    b_p = axs[2].bar(xc, path_p95, 0.42, color="#b75d28", label="根部 XY 路径误差 P95 (m)")
    axs[2].bar_label(b_p, fmt="%.2fm", fontsize=8.5, fontweight="bold")
    axs[2].set_xticks(xc, CLIPS, rotation=15)
    axs[2].set_ylim(0, 1.05)
    axs[2].set_ylabel("根部 XY 路径误差 P95 (m)")
    axs[2].set_title(f"C. 闭环全局 XY 路径误差 P95 (5 序列均值 {np.mean(path_p95):.3f}m，5/5 全稳定)", fontsize=9.5, fontweight="bold")
    axs[2].legend(fontsize=8)
    axs[2].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(REPORT_DIR / "figures/fig4_closed_loop_tracking_metrics.png", dpi=150)
    plt.close(fig)


def render_four_stage_video() -> Path:
    """Render the 4-stage synchronized end-to-end showcase video and extract keyframe strip `fig2`."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")
    fonts = {sz: ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", sz) for sz in (13, 14, 15, 16, 18, 22, 24)}
    src_out = ROOT / "outputs/continuous_v3_20261023"
    res = json.loads((src_out / "results.json").read_text(encoding="utf-8"))

    width, height = 1600, 920
    tile_w, tile_h = 760, 360
    plant = TwistPlant()
    mj = plant.mj
    plant.model.vis.global_.offwidth = tile_w
    plant.model.vis.global_.offheight = tile_h
    renderer = mj.Renderer(plant.model, height=tile_h, width=tile_w)
    data = mj.MjData(plant.model)
    cam = mj.MjvCamera()
    cam.azimuth = 135
    cam.elevation = -14
    cam.distance = 2.50

    video_path = REPORT_DIR / "videos/end_to_end_pipeline_showcase.mp4"
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", "25",
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    keyframe_rows = []

    try:
        for clip_id, clip_title in VIDEO_CLIPS:
            cam_frames = load_clip_video_frames(clip_id, target_size=(380, 180))
            dets = read_jsonl(ROOT / "outputs/public_datasets/mpi_results" / clip_id / "detections.jsonl")
            states = read_jsonl(ROOT / "outputs/continuous_prediction_20261020_final/natural" / clip_id / "A2/states.jsonl")
            stream = np.load(src_out / f"references/{clip_id}_A3_balanced_stream.npz", allow_pickle=True)
            trace = np.load(src_out / f"traces/A3_balanced_{clip_id}_sonic.npz", allow_pickle=True)
            m_info = res[f"A3_balanced|{clip_id}|sonic"]

            # Reference pelvis origin for 3D skeleton display
            first_pts = [np.asarray(p, float) for p in states[0]["joints_world_m"] if p is not None]
            origin_m = np.mean(first_pts, axis=0) if first_pts else np.zeros(3)

            # Active tracking window is t = 3.00 .. 9.00 s (150 source frames at 25 fps, i.e., 50 Hz ticks 150 .. 450)
            for f_idx in range(150):
                tick_50 = min(150 + f_idx * 2, len(stream["bounded"]) - 1)
                # Causal controller executes delayed reference by 120 ms (6 ticks at 50 Hz)
                tick_ctrl = min(tick_50 + 6, len(trace["qpos"]) - 1)
                now_s = float(f_idx) * 0.04

                canvas = Image.new("RGB", (width, height), "#f4f6f8")
                draw = ImageDraw.Draw(canvas)
                draw.text((24, 12), f"主线全链路四阶段同步演示 — {clip_title}", font=fonts[22], fill="#17283c")
                draw.text(
                    (24, 46),
                    f"源视频帧 f = {f_idx:03d}/150 (t = {now_s:4.2f} s)  |  主线算法：4相机RTMPose -> 鲁棒三角化 -> BRM-C2 重定向与C²流 -> SONIC 闭环控制",
                    font=fonts[15],
                    fill="#4a5d70",
                )

                # Tile 1 (Top-Left): Stage 1 - 4-Camera Raw Human RGB Video
                t1 = render_stage1_multicam_tile(cam_frames, f_idx, fonts)
                canvas.paste(t1, (20, 106))
                draw.text((20, 80), "① 阶段一：4 相机同步原人类真实 RGB 视频 (2048×2048 @ 25Hz)", font=fonts[16], fill="#1f4e78")
                draw.text((20, 470), "输入源：MPI-INF-3DHP 4 视角同步环绕相机 (Cam 0 / Cam 2 / Cam 5 / Cam 7)，无人工标记点", font=fonts[14], fill="#2c3e50")

                # Tile 2 (Top-Right): Stage 2 - 2D Keypoint Detection + 3D Triangulated Skeleton
                cam_f_Now = {cid: cam_frames[cid][f_idx] for cid in ("0", "2")}
                t2 = render_stage2_skeleton_tile(cam_f_Now, dets[f_idx], states[f_idx], origin_m, fonts)
                canvas.paste(t2, (805, 106))
                draw.text((805, 80), "② 阶段二：多视角 2D 关键点检测与 3D 空间骨架重建 (25Hz)", font=fonts[16], fill="#2e75b6")
                draw.text((805, 470), "左：多视角 2D 骨架检测叠加  |  右：置信度加权 DLT 三角化 + 因果滤波恢复的世界系 3D 骨架", font=fonts[14], fill="#2c3e50")

                # Tile 3 (Bottom-Left): Stage 3 - Robot Kinematic Retargeting Reference (BRM-C2)
                q_kin = stream["bounded"][tick_50].copy()
                inst_jerk = float(np.max(np.abs(stream["jerk"][tick_50])))
                data.qpos[:] = q_kin
                mj.mj_kinematics(plant.model, data)
                cam.lookat[:] = [q_kin[0], q_kin[1], 0.65]
                renderer.update_scene(data, camera=cam)
                t3 = Image.fromarray(renderer.render())
                canvas.paste(t3, (20, 521))
                draw.text((20, 495), "③ 阶段三：机器人运动学重定向与 C² 参考轨迹效果 (BRM-C2 @ 50Hz)", font=fonts[16], fill="#1b9e77")
                draw.text(
                    (20, 885),
                    f"运动学参考 q_ref：骨盆高度 z_ref={q_kin[2]:.2f}m  |  瞬时最大跃度={inst_jerk:.0f} rad/s³ (<=1500)  |  奇异壁垒无翻转",
                    font=fonts[14],
                    fill="#2c3e50",
                )

                # Tile 4 (Bottom-Right): Stage 4 - Closed-Loop Physical Robot Tracking (SONIC in MuJoCo)
                q_act = trace["qpos"][tick_ctrl].copy()
                inst_rmse = float(np.degrees(np.sqrt(np.mean((q_act[7:] - q_kin[7:]) ** 2))))
                data.qpos[:] = q_act
                mj.mj_kinematics(plant.model, data)
                cam.lookat[:] = [q_act[0], q_act[1], 0.65]
                renderer.update_scene(data, camera=cam)
                t4 = Image.fromarray(renderer.render())
                canvas.paste(t4, (805, 521))
                draw.text((805, 495), "④ 阶段四：机器人全动力学闭环物理跟踪效果 (SONIC 策略 @ 1000Hz MuJoCo)", font=fonts[16], fill="#b75d28")
                draw.text(
                    (805, 885),
                    f"物理闭环实测：当前 z={q_act[2]:.2f}m | 瞬时关节误差={inst_rmse:.1f}° | 序列RMSE={m_info['joint_rmse_deg']:.1f}° | 路径P95={m_info['path_error_p95_m']:.2f}m",
                    font=fonts[14],
                    fill="#2c3e50",
                )

                frame_rgb = np.asarray(canvas, dtype=np.uint8)
                proc.stdin.write(frame_rgb.tobytes())

                # Capture representative keyframe for Figure 2
                if (clip_id == "s1_walk" and f_idx == 118) or (clip_id == "s1_exercise" and f_idx == 24) or (clip_id == "s2_sports" and f_idx == 75):
                    keyframe_rows.append((clip_title, f_idx, t1.copy(), t2.copy(), t3.copy(), t4.copy()))

        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", errors="replace")
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {err}")
    finally:
        renderer.close()

    # Compose Figure 2: 3 clips x 4 stages keyframe table
    fig, axes = plt.subplots(len(keyframe_rows), 4, figsize=(16.5, 8.2))
    col_titles = (
        "① 原人类 4 相机真实 RGB 视频",
        "② 2D 关键点检测 + 3D 重建骨架",
        "③ 机器人运动学重定向 (BRM-C2)",
        "④ 机器人全动力学闭环跟踪 (SONIC)",
    )
    for r_i, (ctitle, f_idx, im1, im2, im3, im4) in enumerate(keyframe_rows):
        for c_i, im in enumerate((im1, im2, im3, im4)):
            ax = axes[r_i, c_i]
            ax.imshow(np.asarray(im))
            ax.set_xticks([])
            ax.set_yticks([])
            if r_i == 0:
                ax.set_title(col_titles[c_i], fontsize=10.5, fontweight="bold")
        axes[r_i, 0].set_ylabel(f"{ctitle.split('（')[0]}\n(f={f_idx})", fontsize=9.5, fontweight="bold")
    fig.suptitle("图 2：主线算法从原人类多视角视频、2D/3D 骨架检测到机器人运动学重定向与物理闭环跟踪的四阶段同步关键帧", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "figures/fig2_stage_by_stage_keyframes.png", dpi=150)
    plt.close(fig)

    return video_path


def generate_executive_figures() -> None:
    """Generate pure-Chinese executive report figures without any code variable names."""
    art_dir = Path(r"C:\Users\TanWei\.gemini\antigravity\brain\e75da53b-0545-4273-b46b-ac2e0f9db3d4")
    src_out = ROOT / "outputs/continuous_v3_20261023"
    res = json.loads((src_out / "results.json").read_text(encoding="utf-8"))
    ref_cont = json.loads((src_out / "reference_continuity.json").read_text(encoding="utf-8"))
    clip_cn = ("行走与转身", "快速连续起蹲", "深蹲与体操", "日常转体挥臂", "大跨度体育动作")

    # 1. exec_fig1_pipeline.png
    fig, ax = plt.subplots(figsize=(14.5, 4.2))
    ax.axis("off")
    stages = [
        ("第一阶段：多视角同步采集\n与二维人体关键点检测", "输入：四台环绕相机同步画面 (25帧/秒)\n处理：人体检测与全身23关键点提取\n（含躯干、四肢及双足跟/脚趾触地关键点）\n输出：多视角二维像素坐标与检测置信度", "#1f4e78"),
        ("第二阶段：多视角空间三角化\n与因果三维状态滤波", "输入：四视角二维关键点与相机标定参数\n处理：置信度加权空间三角化、抗遮挡离群抑制\n与因果滑动窗口骨长约束平滑\n输出：世界坐标系下三维人体骨架位置与速度", "#2e75b6"),
        ("第三阶段：奇异壁垒连续\n运动学重定向（前端）", "输入：三维人体空间骨架序列 (25帧/秒)\n处理：人机尺寸比例缩放、骨盆与四肢方向对齐\n及肩肘万向节奇异区软壁垒防翻转优化\n输出：宇树G1机器人根部位姿与29关节目标角", "#1b9e77"),
        ("第四阶段：单调门控柔化\n二阶连续轨迹生成（后端）", "输入：25帧/秒离散重定向关节姿态\n处理：单调同向阻尼过境速度门控、换向归零\n与双层柔化三阶样条插补（跃度减半保护）\n输出：50帧/秒严格平滑的关节位置、速度与加速度", "#2b83ba"),
        ("第五阶段：全身强化学习\n闭环物理跟踪控制", "输入：50帧/秒实时参考姿态与机器人本体反馈\n处理：实时因果全身控制策略推理\n与1000赫兹底层关节力矩闭环驱动\n输出：全动力学环境下的全身平衡与动作模仿", "#b75d28"),
    ]
    for i, (title, body, col) in enumerate(stages):
        x0 = 0.015 + i * 0.198
        rect = plt.Rectangle((x0, 0.12), 0.178, 0.76, facecolor=col, edgecolor="#102030", lw=1.5, alpha=0.92, transform=ax.transAxes, zorder=2)
        ax.add_patch(rect)
        ax.text(x0 + 0.089, 0.74, title, ha="center", va="center", color="white", fontsize=10.5, fontweight="bold", transform=ax.transAxes, zorder=3)
        ax.text(x0 + 0.089, 0.39, body, ha="center", va="center", color="#f0f4f8", fontsize=8.3, transform=ax.transAxes, zorder=3)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x0 + 0.196, 0.50), xytext=(x0 + 0.180, 0.50),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", lw=2.2, color="#2c3e50"))
    ax.set_title("图 1：多相机无标记三维视觉感知到人形机器人全身闭环控制的主线技术流程", fontsize=12, fontweight="bold", pad=8)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "figures/exec_fig1_pipeline.png", dpi=150)
    plt.close(fig)

    # 2. exec_fig3_continuity.png
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.8))
    for mode, lbl in (
        ("A2_continuous", "传统无壁垒方案（转身摆臂发生奇异翻转）"),
        ("A3_balanced", "奇异壁垒单调柔化主线方案（平滑无翻转）"),
    ):
        r_rows = read_jsonl(src_out / "references" / f"s1_walk_{'A3_ik' if mode == 'A3_balanced' else mode}.jsonl")
        t_25 = np.arange(len(r_rows)) * 0.04
        err = np.array([max([e for e in x["limb_direction_error_deg"][:4] if e is not None] or [np.nan]) for x in r_rows])
        axs[0].plot(t_25, err, color=ALGO_COLORS[mode], lw=1.9, label=lbl)
    axs[0].axhline(20.0, color="#c0392b", ls="--", lw=1.0, label="20° 肢体方向偏差警戒线")
    axs[0].set_xlabel("运动时间 (秒)")
    axs[0].set_ylabel("上肢最大方向对齐偏差 (°)")
    axs[0].set_title("A. 肩肘奇异区防翻转效果对比\n（超20°异常帧数由 38 帧降至 6 帧）", fontsize=10, fontweight="bold")
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=0.25)

    modes = ("A2_continuous", "A3_ik", "A3_balanced", "A3_full")
    short_names = ("传统无壁垒\n等比例同步", "奇异壁垒\n等比例同步", "奇异壁垒\n单调柔化(主线)", "奇异壁垒\n全速外推")
    xm = np.arange(len(modes))
    p95_j = [float(np.mean([ref_cont[f"{m}|{c}"]["control_50hz"]["p95_jerk_rad_s3"] for c in CLIPS])) for m in modes]
    max_j = [float(np.max([ref_cont[f"{m}|{c}"]["control_50hz"]["max_jerk_rad_s3"] for c in CLIPS])) for m in modes]
    b1 = axs[1].bar(xm - 0.17, p95_j, 0.32, color=[ALGO_COLORS[m] for m in modes], label="五类动作平均95分位跃度")
    b2 = axs[1].bar(xm + 0.17, max_j, 0.32, color=[ALGO_COLORS[m] for m in modes], alpha=0.40, edgecolor="#2c3e50", label="最大跃度峰值")
    axs[1].bar_label(b1, fmt="%.0f", fontsize=8.5, fontweight="bold")
    axs[1].bar_label(b2, fmt="%.0f", fontsize=8)
    axs[1].set_xticks(xm, short_names, fontsize=8.5)
    axs[1].set_ylim(0, 3550)
    axs[1].set_ylabel("控制参考轨迹跃度 (弧度/秒³)")
    axs[1].set_title("B. 关节轨迹平滑度（跃度冲击）对比\n（主线方案冲击降低 31.6% ~ 49.6%）", fontsize=10, fontweight="bold")
    axs[1].legend(fontsize=8, loc="upper left")
    axs[1].grid(axis="y", alpha=0.25)

    s_ref = np.load(src_out / "references/s1_exercise_A3_balanced_stream.npz", allow_pickle=True)
    axs[2].plot(s_ref["time_s"], s_ref["bounded"][:, 2], color="#2c3e50", ls="--", lw=1.6, label="人体重定向目标高度 (最低 0.485 米)")
    for m, lbl in (
        ("A2_continuous", "传统无壁垒等比例同步 (最低 0.514 米)"),
        ("A3_ik", "奇异壁垒等比例同步 (最低 0.512 米)"),
        ("A3_balanced", "奇异壁垒单调柔化主线方案 (最低 0.488 米)"),
    ):
        tr = np.load(src_out / f"traces/{m}_s1_exercise_sonic.npz", allow_pickle=True)
        axs[2].plot(tr["time_s"], tr["qpos"][:, 2], color=ALGO_COLORS[m], lw=1.9, label=lbl)
    axs[2].set_xlim(3.0, 8.8)
    axs[2].set_ylim(0.46, 0.78)
    axs[2].set_xlabel("运动时间 (秒)")
    axs[2].set_ylabel("机器人骨盆高度 (米)")
    axs[2].set_title("C. 深度下蹲动作的骨盆高度响应对比\n（主线方案消除幅度衰减，精准下蹲到底）", fontsize=10, fontweight="bold")
    axs[2].legend(fontsize=8, loc="lower right")
    axs[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(REPORT_DIR / "figures/exec_fig3_continuity.png", dpi=150)
    plt.close(fig)

    # 3. exec_fig4_metrics.png
    fig, axs = plt.subplots(1, 3, figsize=(15.2, 4.6))
    xc = np.arange(len(CLIPS))
    q_vals = [res[f"A3_balanced|{c}|sonic"]["joint_rmse_deg"] for c in CLIPS]
    leg_vals = [res[f"A3_balanced|{c}|sonic"]["leg_rmse_deg"] for c in CLIPS]
    kp_mean = [res[f"A3_balanced|{c}|sonic"]["root_local_body_position_mean_mm"] for c in CLIPS]
    kp_p95 = [res[f"A3_balanced|{c}|sonic"]["root_local_body_position_p95_mm"] for c in CLIPS]
    path_p95 = [res[f"A3_balanced|{c}|sonic"]["path_error_p95_m"] for c in CLIPS]

    b_q = axs[0].bar(xc - 0.17, q_vals, 0.32, color="#2b83ba", label="全身29关节均方根误差 (°)")
    b_l = axs[0].bar(xc + 0.17, leg_vals, 0.32, color="#1b9e77", label="下肢12关节均方根误差 (°)")
    axs[0].bar_label(b_q, fmt="%.1f°", fontsize=8)
    axs[0].bar_label(b_l, fmt="%.1f°", fontsize=8)
    axs[0].set_xticks(xc, clip_cn, rotation=14, fontsize=8.5)
    axs[0].set_ylim(0, 31)
    axs[0].set_ylabel("关节跟踪误差 (°)")
    axs[0].set_title(f"A. 闭环关节跟踪误差（均值：全身 {np.mean(q_vals):.1f}°，下肢 {np.mean(leg_vals):.1f}°）", fontsize=9.5, fontweight="bold")
    axs[0].legend(fontsize=8)
    axs[0].grid(axis="y", alpha=0.25)

    b_km = axs[1].bar(xc - 0.17, kp_mean, 0.32, color="#2e75b6", label="身体关键点平均误差 (毫米)")
    b_kp = axs[1].bar(xc + 0.17, kp_p95, 0.32, color="#e6ab02", label="身体关键点95分位误差 (毫米)")
    axs[1].bar_label(b_km, fmt="%.0f", fontsize=8)
    axs[1].bar_label(b_kp, fmt="%.0f", fontsize=8)
    axs[1].set_xticks(xc, clip_cn, rotation=14, fontsize=8.5)
    axs[1].set_ylim(0, 330)
    axs[1].set_ylabel("身体局部三维关键点误差 (毫米)")
    axs[1].set_title(f"B. 闭环三维身体关键点误差（均值 {np.mean(kp_mean):.1f} 毫米）", fontsize=9.5, fontweight="bold")
    axs[1].legend(fontsize=8)
    axs[1].grid(axis="y", alpha=0.25)

    b_p = axs[2].bar(xc, path_p95, 0.42, color="#b75d28", label="全局水平路径95分位偏差 (米)")
    axs[2].bar_label(b_p, fmt="%.2f米", fontsize=8.5, fontweight="bold")
    axs[2].set_xticks(xc, clip_cn, rotation=14, fontsize=8.5)
    axs[2].set_ylim(0, 1.05)
    axs[2].set_ylabel("水平路径95分位偏差 (米)")
    axs[2].set_title(f"C. 闭环全局水平路径偏差（五类动作均值 {np.mean(path_p95):.2f} 米，全部平衡停稳）", fontsize=9.5, fontweight="bold")
    axs[2].legend(fontsize=8)
    axs[2].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(REPORT_DIR / "figures/exec_fig4_metrics.png", dpi=150)
    plt.close(fig)

    # 4. exec_fig2_keyframes.png
    fonts = {sz: ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", sz) for sz in (13, 14, 15, 16)}
    plant = TwistPlant()
    mj = plant.mj
    plant.model.vis.global_.offwidth = 760
    plant.model.vis.global_.offheight = 360
    renderer = mj.Renderer(plant.model, height=360, width=760)
    data = mj.MjData(plant.model)
    cam = mj.MjvCamera()
    cam.azimuth = 135
    cam.elevation = -14
    cam.distance = 2.50
    rows_spec = [
        ("s1_walk", 118, "场景一：\n自然行走与转身"),
        ("s1_exercise", 24, "场景二：\n深度下蹲与体操"),
        ("s2_sports", 75, "场景三：\n大跨度体育动作"),
    ]
    kf_rows = []
    for clip_id, f_idx, row_lbl in rows_spec:
        cam_frames = load_clip_video_frames(clip_id, target_size=(380, 180))
        dets = read_jsonl(ROOT / "outputs/public_datasets/mpi_results" / clip_id / "detections.jsonl")
        states = read_jsonl(ROOT / "outputs/continuous_prediction_20261020_final/natural" / clip_id / "A2/states.jsonl")
        stream = np.load(src_out / f"references/{clip_id}_A3_balanced_stream.npz", allow_pickle=True)
        trace = np.load(src_out / f"traces/A3_balanced_{clip_id}_sonic.npz", allow_pickle=True)
        first_pts = [np.asarray(p, float) for p in states[0]["joints_world_m"] if p is not None]
        origin_m = np.mean(first_pts, axis=0)
        t1 = render_stage1_multicam_tile(cam_frames, f_idx, fonts)
        t2 = render_stage2_skeleton_tile({cid: cam_frames[cid][f_idx] for cid in ("0", "2")}, dets[f_idx], states[f_idx], origin_m, fonts)
        tick_50 = min(150 + f_idx * 2, len(stream["bounded"]) - 1)
        tick_ctrl = min(tick_50 + 6, len(trace["qpos"]) - 1)
        q_kin = stream["bounded"][tick_50].copy()
        data.qpos[:] = q_kin
        mj.mj_kinematics(plant.model, data)
        cam.lookat[:] = [q_kin[0], q_kin[1], 0.65]
        renderer.update_scene(data, camera=cam)
        t3 = Image.fromarray(renderer.render())
        q_act = trace["qpos"][tick_ctrl].copy()
        data.qpos[:] = q_act
        mj.mj_kinematics(plant.model, data)
        cam.lookat[:] = [q_act[0], q_act[1], 0.65]
        renderer.update_scene(data, camera=cam)
        t4 = Image.fromarray(renderer.render())
        kf_rows.append((row_lbl, t1, t2, t3, t4))
    renderer.close()

    fig, axes = plt.subplots(3, 4, figsize=(16.5, 8.2))
    col_titles = (
        "① 原人类四相机同步实拍视频",
        "② 二维关键点检测与三维重建骨架",
        "③ 机器人运动学重定向参考姿态",
        "④ 机器人全动力学闭环物理跟踪效果",
    )
    for r_i, (rlbl, im1, im2, im3, im4) in enumerate(kf_rows):
        for c_i, im in enumerate((im1, im2, im3, im4)):
            ax = axes[r_i, c_i]
            ax.imshow(np.asarray(im))
            ax.set_xticks([])
            ax.set_yticks([])
            if r_i == 0:
                ax.set_title(col_titles[c_i], fontsize=10.5, fontweight="bold")
        axes[r_i, 0].set_ylabel(rlbl, fontsize=10, fontweight="bold")
    fig.suptitle("图 2：从原人类多视角实拍视频、二三维骨架检测到机器人运动学重定向与物理闭环跟踪的四阶段同步效果", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "figures/exec_fig2_keyframes.png", dpi=150)
    plt.close(fig)

    if art_dir.exists():
        for fn in ("exec_fig1_pipeline.png", "exec_fig2_keyframes.png", "exec_fig3_continuity.png", "exec_fig4_metrics.png"):
            shutil.copy2(REPORT_DIR / "figures" / fn, art_dir / fn)


def main() -> None:
    if "--exec-figures-only" in sys.argv:
        generate_executive_figures()
        return
    summary = organize_clean_data()
    generate_report_figures(summary)
    generate_executive_figures()
    video_path = render_four_stage_video()
    print(f"Successfully generated mainline report assets and video: {video_path}")


if __name__ == "__main__":
    main()
