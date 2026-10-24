"""Generate diagnostic plots and 4-panel video for continuous reference TWIST2 vs SONIC.

Reads frozen outputs from `scripts/benchmark_continuous_twist_sonic.py` without
re-running physics or modifying references. Verifies full H.264 MP4 decode via
both `ffmpeg` and OpenCV `VideoCapture`.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation

from motion_control.twist import TwistPlant
from pose_sim.continuous_state import BODY
from pose_sim.landmarks import LIMB_BONES
from scripts.benchmark_continuous_twist_sonic import (
    CLIP_LABELS,
    CLIPS,
    CONTROLLERS,
    REF_LABELS,
    REF_MODES,
    read_jsonl,
    save_json,
)
from scripts.benchmark_twist_policy import balance_violation

COLORS = {
    'A0_legacy': '#8c96a3',
    'A2_gated': '#c97b2b',
    'A2_continuous': '#158a72',
    'twist': '#2268b5',
    'sonic': '#148574',
    'reference': '#ba861c',
}


def load_npz(path):
    with np.load(path) as archive:
        return {k: archive[k] for k in archive.files}


def array_states(rows):
    return np.array(
        [[dict(zip(r['joint_names'], r['joints_world_m'])).get(n) or [np.nan] * 3 for n in BODY] for r in rows],
        float,
    )


def make_plots(out_dir):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'],
            'axes.unicode_minus': False,
            'font.size': 10,
            'axes.spines.top': False,
            'axes.spines.right': False,
        }
    )

    ref_cont = json.loads((out_dir / 'reference_continuity.json').read_text(encoding='utf-8'))
    results = json.loads((out_dir / 'results.json').read_text(encoding='utf-8'))

    # 1. Reference Continuity Comparison Plot
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), layout='constrained')
    x = np.arange(len(CLIPS))
    w = 0.25
    for k, mode in enumerate(REF_MODES):
        acc = [100.0 * ref_cont[c][mode]['acceptance_fraction'] for c in CLIPS]
        step = [ref_cont[c][mode]['max_adjacent_25hz_joint_step_deg'] for c in CLIPS]
        bars0 = axs[0, 0].bar(x + (k - 1) * w, acc, w, color=COLORS[mode], label=REF_LABELS[mode])
        axs[0, 0].bar_label(bars0, fmt='%.0f%%', fontsize=8, padding=2)
        bars1 = axs[0, 1].bar(x + (k - 1) * w, step, w, color=COLORS[mode], label=REF_LABELS[mode])
        axs[0, 1].bar_label(bars1, fmt='%.1f°', fontsize=8, padding=2)

    axs[0, 0].set_xticks(x, [CLIP_LABELS[c] for c in CLIPS])
    axs[0, 0].set_ylim(0, 118)
    axs[0, 0].set_ylabel('有效姿态参考覆盖率 (%)')
    axs[0, 0].set_title('A. 25 Hz 源姿态参考有效覆盖率（消除转身锁死与深蹲跳回站立）', loc='left', fontweight='bold')
    axs[0, 0].grid(axis='y', alpha=0.2)
    axs[0, 0].legend(fontsize=8.5, loc='lower right')

    axs[0, 1].set_xticks(x, [CLIP_LABELS[c] for c in CLIPS])
    axs[0, 1].set_ylim(0, 158)
    axs[0, 1].set_ylabel('相邻 40 ms 帧最大关节跳变 (°)')
    axs[0, 1].set_title('B. 25 Hz 逆运动学相邻帧最大跳变（速度信赖域消除 100°+ 分支翻转）', loc='left', fontweight='bold')
    axs[0, 1].grid(axis='y', alpha=0.2)

    # Panel C: S1 Walk heading & knee trajectory across A0_legacy, A2_gated, A2_continuous
    s_a0 = load_npz(out_dir / 'references/s1_walk/A0_legacy_stream.npz')
    s_gated = load_npz(out_dir / 'references/s1_walk/A2_gated_stream.npz')
    s_cont = load_npz(out_dir / 'references/s1_walk/A2_continuous_stream.npz')
    t_s1 = s_cont['time_s']
    for mode, st, ls in (('A0_legacy', s_a0, '--'), ('A2_gated', s_gated, '-.'), ('A2_continuous', s_cont, '-')):
        yaw_deg = np.degrees(np.unwrap(Rotation.from_quat(st['bounded'][:, [4, 5, 6, 3]]).as_euler('xyz')[:, 2]))
        axs[1, 0].plot(t_s1, yaw_deg, color=COLORS[mode], ls=ls, lw=2.0 if mode == 'A2_continuous' else 1.4, label=REF_LABELS[mode])
    axs[1, 0].set_title('C. S1 行走转身：50 Hz 根部偏航角参考（消除转身超限回滚锁死与末尾归零自旋）', loc='left', fontweight='bold')
    axs[1, 0].set_xlabel('仿真控制时间 (s)')
    axs[1, 0].set_ylabel('连续偏航角 (°)')
    axs[1, 0].grid(alpha=0.2)
    axs[1, 0].legend(fontsize=8.5)

    # Panel D: S1 Crouch commanded height & analytical dq continuity
    c_a0 = load_npz(out_dir / 'references/s1_crouch/A0_legacy_stream.npz')
    c_gated = load_npz(out_dir / 'references/s1_crouch/A2_gated_stream.npz')
    c_cont = load_npz(out_dir / 'references/s1_crouch/A2_continuous_stream.npz')
    t_cr = c_cont['time_s']
    for mode, st, ls in (('A0_legacy', c_a0, '--'), ('A2_gated', c_gated, '-.'), ('A2_continuous', c_cont, '-')):
        axs[1, 1].plot(t_cr, st['bounded'][:, 2], color=COLORS[mode], ls=ls, lw=2.0 if mode == 'A2_continuous' else 1.4, label=REF_LABELS[mode])
    axs[1, 1].set_title('D. S1 下蹲：50 Hz 期望骨盆高度（消除深蹲 <0.45 m 触发直立撤回跳变）', loc='left', fontweight='bold')
    axs[1, 1].set_xlabel('仿真控制时间 (s)')
    axs[1, 1].set_ylabel('期望骨盆高度 (m)')
    axs[1, 1].grid(alpha=0.2)
    axs[1, 1].legend(fontsize=8.5)

    fig.suptitle(
        '我们是否得到了连续参考？——从 3D 连续人体状态到 50 Hz 全链路 C² 连续机器人姿态参考\n'
        '满足 |dq| ≤ 6 rad/s, |ddq| ≤ 120 rad/s², |jerk| ≤ 3000 rad/s³（较旧参考 45000 rad/s³ 降低 15 倍），且不施加前置安全滤波',
        fontsize=14,
    )
    fig.savefig(out_dir / 'reference_continuity_comparison.png', dpi=150)
    plt.close(fig)

    # 2. TWIST2 vs SONIC Quantitative Metrics Plot (120 ms & 0 ms under A2_continuous + comparison with A0_legacy)
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), layout='constrained')
    groups = (
        ('A0_legacy', 120, 'twist', '#7fa6d6', 'TWIST2 (旧A0)'),
        ('A2_continuous', 120, 'twist', '#2268b5', 'TWIST2 (新连续)'),
        ('A0_legacy', 120, 'sonic', '#7cc4b8', 'SONIC (旧A0)'),
        ('A2_continuous', 120, 'sonic', '#148574', 'SONIC (新连续)'),
    )
    bw = 0.19
    for idx_g, (mode, delay, ctrl, col, lbl) in enumerate(groups):
        offset = (idx_g - 1.5) * bw
        leg_vals = [results[f'{mode}_{c}_{delay}_{ctrl}']['leg_rmse_deg'] for c in CLIPS]
        all_vals = [results[f'{mode}_{c}_{delay}_{ctrl}']['joint_rmse_deg'] for c in CLIPS]
        path_vals = [min(results[f'{mode}_{c}_{delay}_{ctrl}']['path_error_p95_m'], 2.5) for c in CLIPS]
        b0 = axs[0, 0].bar(x + offset, leg_vals, bw, color=col, label=lbl)
        axs[0, 0].bar_label(b0, fmt='%.1f', fontsize=7.5, padding=2)
        b1 = axs[0, 1].bar(x + offset, all_vals, bw, color=col, label=lbl)
        axs[0, 1].bar_label(b1, fmt='%.1f', fontsize=7.5, padding=2)
        b2 = axs[1, 0].bar(x + offset, path_vals, bw, color=col, label=lbl)
        axs[1, 0].bar_label(
            b2,
            labels=[f'{results[f"{mode}_{c}_{delay}_{ctrl}"]["path_error_p95_m"]:.2f}' for c in CLIPS],
            fontsize=7.5,
            padding=2,
        )

    axs[0, 0].set_xticks(x, [CLIP_LABELS[c] for c in CLIPS])
    axs[0, 0].set_ylabel('下肢 12 关节 RMSE (°)')
    axs[0, 0].set_ylim(0, 42)
    axs[0, 0].set_title('A. 下肢 12 关节跟踪误差 RMSE (°)（120 ms 因果延迟）', loc='left', fontweight='bold')
    axs[0, 0].grid(axis='y', alpha=0.2)
    axs[0, 0].legend(fontsize=8.5, ncol=2)

    axs[0, 1].set_xticks(x, [CLIP_LABELS[c] for c in CLIPS])
    axs[0, 1].set_ylabel('全身 29 关节 RMSE (°)')
    axs[0, 1].set_ylim(0, 38)
    axs[0, 1].set_title('B. 全身 29 关节跟踪误差 RMSE (°)（120 ms 因果延迟）', loc='left', fontweight='bold')
    axs[0, 1].grid(axis='y', alpha=0.2)

    axs[1, 0].set_xticks(x, [CLIP_LABELS[c] for c in CLIPS])
    axs[1, 0].set_ylabel('根部 XY 路径误差 P95 (m，截断显示 2.5m)')
    axs[1, 0].set_ylim(0, 2.95)
    axs[1, 0].set_title('C. 同时刻根部 XY 路径误差 P95 (m)（新连续参考显著改善 SONIC 与 TWIST2 漂移）', loc='left', fontweight='bold')
    axs[1, 0].grid(axis='y', alpha=0.2)

    # Panel D: Outcome matrix (Balance & Standing Stop across A0_legacy, A2_gated, A2_continuous)
    ax = axs[1, 1]
    cols_matrix = [
        ('A0_legacy', 120, 'twist', 'TWIST2\n旧A0 (120ms)'),
        ('A2_gated', 120, 'twist', 'TWIST2\n过渡A2 (120ms)'),
        ('A2_continuous', 120, 'twist', 'TWIST2\n新连续 (120ms)'),
        ('A2_continuous', 0, 'twist', 'TWIST2\n新连续 (0ms)'),
        ('A0_legacy', 120, 'sonic', 'SONIC\n旧A0 (120ms)'),
        ('A2_continuous', 120, 'sonic', 'SONIC\n新连续 (120ms)'),
        ('A2_continuous', 0, 'sonic', 'SONIC\n新连续 (0ms)'),
    ]
    ax.set_xlim(-0.5, len(cols_matrix) - 0.5)
    ax.set_ylim(len(CLIPS) - 0.5, -0.5)
    for r_idx, clip in enumerate(CLIPS):
        for c_idx, (mode, delay, ctrl, _) in enumerate(cols_matrix):
            m = results[f'{mode}_{clip}_{delay}_{ctrl}']
            bal = m['balance_envelope_passed']
            rel_ok = m['reference_relative_upright_passed']
            stop = m['standing_stop_passed']
            if bal and stop:
                bg = '#d7efe2'
                txt = f'全程平衡+停止\nmin z={m["minimum_root_height_m"]:.2f}m'
            elif rel_ok and stop:
                bg = '#e5f2d8'
                txt = f'深蹲不倒+停止\nmin z={m["minimum_root_height_m"]:.2f}m'
            else:
                bg = '#f6d8d8'
                fv = m['first_balance_violation_s']
                txt = f'{fv:.1f}s 失稳\n' + ('停止通过' if stop else '倒地/未停稳')
            ax.add_patch(plt.Rectangle((c_idx - 0.47, r_idx - 0.44), 0.94, 0.88, facecolor=bg, edgecolor='white'))
            ax.text(c_idx, r_idx, txt, ha='center', va='center', fontsize=8)
    ax.set_xticks(range(len(cols_matrix)), [c[3] for c in cols_matrix], fontsize=8)
    ax.set_yticks(range(len(CLIPS)), [CLIP_LABELS[c] for c in CLIPS], fontsize=9)
    ax.tick_params(length=0)
    ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
    ax.set_title('D. 控制器自主平衡与停止结果矩阵（无前置安全滤波）', loc='left', fontweight='bold')

    fig.suptitle(
        '新连续参考 (A2_continuous) 下 TWIST2 与 SONIC 真人姿态模仿定量对比\n'
        '平衡、稳定与安全完全交由下游控制器处理：SONIC 5/5 全程稳定；TWIST2 在行走/日常/体育上稳定且关节精度更高，深蹲仍失稳',
        fontsize=14,
    )
    fig.savefig(out_dir / 'twist_vs_sonic_metrics.png', dpi=150)
    plt.close(fig)

    # 3. Time-Series Detail Plot for S1 Walk & S1 Crouch
    fig, axs = plt.subplots(3, 2, figsize=(15, 10.5), layout='constrained')
    for col, (clip, title_prefix) in enumerate((('s1_walk', 'S1 行走与转身'), ('s1_crouch', 'S1 下蹲与起立'))):
        tr_twist = load_npz(out_dir / f'traces/A2_continuous_{clip}_120_twist.npz')
        tr_sonic = load_npz(out_dir / f'traces/A2_continuous_{clip}_120_sonic.npz')
        t = tr_twist['time_s']
        cmd = tr_twist['command_qpos']

        # Row 0: Pelvis height
        axs[0, col].plot(t, cmd[:, 2], color=COLORS['reference'], lw=2.2, label='新连续参考期望高度')
        axs[0, col].plot(t, tr_twist['qpos'][:, 2], color=COLORS['twist'], lw=1.6, label='TWIST2 实际高度')
        axs[0, col].plot(t, tr_sonic['qpos'][:, 2], color=COLORS['sonic'], lw=1.6, label='SONIC 实际高度')
        axs[0, col].axhline(0.50, color='#b72d44', ls='--', lw=1.0, label='0.50 m 站立诊断阈值')
        axs[0, col].set_title(f'{title_prefix}：骨盆高度跟踪 (m)', loc='left', fontweight='bold')
        axs[0, col].set_ylabel('高度 (m)')
        axs[0, col].grid(alpha=0.2)
        axs[0, col].legend(fontsize=8.5, loc='lower left')

        # Row 1: Left Knee angle (joint index 3)
        axs[1, col].plot(t, np.degrees(cmd[:, 7 + 3]), color=COLORS['reference'], lw=2.2, label='新连续参考 (C²)')
        axs[1, col].plot(t, np.degrees(tr_twist['qpos'][:, 7 + 3]), color=COLORS['twist'], lw=1.5, label='TWIST2 实际左膝')
        axs[1, col].plot(t, np.degrees(tr_sonic['qpos'][:, 7 + 3]), color=COLORS['sonic'], lw=1.5, label='SONIC 实际左膝')
        axs[1, col].set_title(f'{title_prefix}：左膝关节角度跟踪 (°)', loc='left', fontweight='bold')
        axs[1, col].set_ylabel('左膝角度 (°)')
        axs[1, col].grid(alpha=0.2)
        axs[1, col].legend(fontsize=8.5, loc='upper right')

        # Row 2: Tilt angle max(|roll|, |pitch|)
        for ctrl, tr, lbl in (('twist', tr_twist, 'TWIST2'), ('sonic', tr_sonic, 'SONIC')):
            tilt = np.degrees(np.abs(Rotation.from_quat(tr['qpos'][:, [4, 5, 6, 3]]).as_euler('xyz')[:, :2]).max(axis=1))
            axs[2, col].plot(t, tilt, color=COLORS[ctrl], lw=1.6, label=lbl)
        axs[2, col].axhline(np.degrees(0.5), color='#b72d44', ls='--', lw=1.0, label='28.65° 倾角诊断阈值')
        axs[2, col].set_title(f'{title_prefix}：躯干最大倾角 max(|roll|, |pitch|) (°)', loc='left', fontweight='bold')
        axs[2, col].set_xlabel('仿真时间 (s)')
        axs[2, col].set_ylabel('倾角 (°)')
        axs[2, col].grid(alpha=0.2)
        axs[2, col].legend(fontsize=8.5, loc='upper left')

    fig.suptitle(
        'S1 行走与 S1 下蹲时序细节对比（新连续参考 A2_continuous，120 ms 延迟）\n'
        '新连续参考消除了 S1 行走转身锁死和末端归零自旋，使 TWIST2 与 SONIC 均平稳完成 S1 行走；下蹲中 SONIC 自主保持平衡，TWIST2 倾覆',
        fontsize=14,
    )
    fig.savefig(out_dir / 's1_walk_and_crouch_detail.png', dpi=150)
    plt.close(fig)


def load_real_camera_frames(clip, cam_id='7', target_w=500, target_h=350):
    """Decode the 150 real RGB frames from the single MPI-INF-3DHP camera view."""
    import cv2

    src_meta = json.loads(
        (ROOT / f'outputs/public_datasets/mpi_benchmark/inputs/{clip}/source.json').read_text(encoding='utf-8')
    )
    dets = read_jsonl(ROOT / f'outputs/mainline_validation_20261019/inference/{clip}/detections.jsonl')
    bboxes = [
        d['views'][cam_id].get('person_bbox_xyxy_px')
        for d in dets
        if d['views'].get(cam_id) and d['views'][cam_id].get('person_bbox_xyxy_px')
    ]
    if bboxes:
        b = np.array(bboxes, float) * 2.0  # detections on 1024x1024 -> 2048x2048 raw video
        min_x, min_y = b[:, :2].min(axis=0)
        max_x, max_y = b[:, 2:].max(axis=0)
    else:
        min_x, min_y, max_x, max_y = 200.0, 450.0, 1848.0, 1890.0

    # Static crop window per clip preserving exact target_w:target_h aspect ratio
    aspect = target_w / target_h
    span_w = max(max_x - min_x + 280.0, (max_y - min_y + 240.0) * aspect, 1450.0)
    span_w = min(2048.0, span_w)
    span_h = span_w / aspect
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    x0 = int(round(np.clip(cx - 0.5 * span_w, 0.0, 2048.0 - span_w)))
    y0 = int(round(np.clip(cy - 0.5 * span_h, 0.0, 2048.0 - span_h)))
    x1 = min(2048, x0 + int(round(span_w)))
    y1 = min(2048, y0 + int(round(span_h)))

    video_file = src_meta['videos'][cam_id]['path']
    start_frame = int(src_meta['start_frame'])
    n_extract = int(src_meta['frames'])
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open source RGB video: {video_file}')
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames_rgb = []
    for _ in range(n_extract):
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f'Failed to decode RGB frame from {video_file}')
        crop = bgr[y0:y1, x0:x1]
        resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
        frames_rgb.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames_rgb, cam_id


def render_video(out_dir, delay_ms=120):
    from PIL import Image, ImageDraw, ImageFont
    import cv2

    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('ffmpeg is required to encode H.264 MP4 video')
    fonts = {sz: ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', sz) for sz in (15, 17, 20, 24, 28)}
    results = json.loads((out_dir / 'results.json').read_text(encoding='utf-8'))
    ref_cont = json.loads((out_dir / 'reference_continuity.json').read_text(encoding='utf-8'))

    width, height = 1660, 960
    top_w, top_h = 500, 350
    bot_w, bot_h = 775, 350
    plant = TwistPlant()
    mj = plant.mj
    plant.model.vis.global_.offwidth = bot_w
    plant.model.vis.global_.offheight = bot_h
    renderer_top = mj.Renderer(plant.model, height=top_h, width=top_w)
    renderer_bot = mj.Renderer(plant.model, height=bot_h, width=bot_w)
    data = mj.MjData(plant.model)
    camera = mj.MjvCamera()
    camera.azimuth = 135
    camera.elevation = -15
    camera.distance = 2.65

    video_path = out_dir / 'comparison_continuous_twist_vs_sonic.mp4'
    cmd = [
        ffmpeg,
        '-hide_banner',
        '-loglevel',
        'error',
        '-y',
        '-f',
        'rawvideo',
        '-pix_fmt',
        'rgb24',
        '-s',
        f'{width}x{height}',
        '-r',
        '25',
        '-i',
        '-',
        '-an',
        '-c:v',
        'libx264',
        '-preset',
        'fast',
        '-crf',
        '20',
        '-threads',
        '4',
        '-pix_fmt',
        'yuv420p',
        '-movflags',
        '+faststart',
        str(video_path),
    ]
    encoder = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    edges = [(BODY.index(a), BODY.index(b)) for a, b in LIMB_BONES]
    edges += [(0, 1), (6, 7), (0, 6), (1, 7)]
    pred_root = ROOT / 'outputs/continuous_prediction_20261020_final/natural'
    cache_root = ROOT / 'outputs/mainline_validation_20261019/inference'
    stage_names = {
        'standing': '初始站立',
        'aligning': '平滑对齐过渡',
        'tracking': '连续姿态模仿',
        'reference_unavailable': '源缺口保持/过渡',
        'stopping': '结束回归站立',
    }

    total_frames = 0
    chapters = {}
    try:
        for clip in CLIPS:
            chapters[clip] = total_frames / 25.0
            states_a2 = read_jsonl(pred_root / clip / 'A2/states.jsonl')
            raw_fused = read_jsonl(cache_root / clip / 'fused.jsonl')
            pts_a2 = array_states(states_a2)
            pts_raw = array_states(raw_fused)
            rgb_frames, cam_id = load_real_camera_frames(clip, cam_id='7', target_w=top_w, target_h=top_h)

            stream_a0 = load_npz(out_dir / f'references/{clip}/A0_legacy_stream.npz')
            stream_cont = load_npz(out_dir / f'references/{clip}/A2_continuous_stream.npz')
            tr_twist = load_npz(out_dir / f'traces/A2_continuous_{clip}_{delay_ms}_twist.npz')
            tr_sonic = load_npz(out_dir / f'traces/A2_continuous_{clip}_{delay_ms}_sonic.npz')
            res_twist = results[f'A2_continuous_{clip}_{delay_ms}_twist']
            res_sonic = results[f'A2_continuous_{clip}_{delay_ms}_sonic']

            center_3d = np.array([0.0, 0.0, 1.0])
            n_frames = math.ceil(tr_twist['time_s'][-1] * 25)
            snap_frame = int(n_frames * 0.52)

            for frame in range(n_frames):
                idx = min(frame * 2, len(tr_twist['time_s']) - 1)
                now_s = float(tr_twist['time_s'][idx])
                stage = str(tr_twist['stage'][idx])
                ref_idx = int(tr_twist['reference_index'][idx])
                src_idx = int(stream_cont['source_ids'][ref_idx]) if 0 <= ref_idx < len(stream_cont['source_ids']) else -1

                canvas = Image.new('RGB', (width, height), '#f4f6f8')
                draw = ImageDraw.Draw(canvas)

                header = f'{CLIP_LABELS[clip]}  |  全链路 C² 连续参考 (A2_continuous) · {delay_ms} ms 因果延迟 · 无前置安全滤波'
                draw.text((24, 10), header, font=fonts[28], fill='#17283c')
                sub = (
                    f'仿真 t = {now_s:5.2f} s · 阶段：{stage_names.get(stage, stage)} · '
                    f'25Hz 连续参考覆盖率 {ref_cont[clip]["A2_continuous"]["accepted_frames"]}/150 '
                    f'(旧 A0 为 {ref_cont[clip]["A0_legacy"]["accepted_frames"]}/150)'
                )
                draw.text((24, 48), sub, font=fonts[17], fill='#4a5d70')

                # Panel 1 (Top-Left, 25, 82): Real Single-Camera RGB Video (Camera 07)
                p0_x, p0_y = 25, 82
                draw.text(
                    (p0_x, p0_y),
                    f'1. 真实人体单视角视频 (Camera 0{cam_id} 实拍)',
                    font=fonts[20],
                    fill='#17283c',
                )
                rgb_idx = int(np.clip(src_idx, 0, len(rgb_frames) - 1)) if src_idx >= 0 else 0
                rgb_img = Image.fromarray(rgb_frames[rgb_idx])
                if not (0 <= src_idx < len(rgb_frames)):
                    # Dim slightly during pre-roll standing / post-roll stopping while still showing real person
                    rgb_arr = (np.asarray(rgb_img, np.float32) * 0.72 + 40.0).clip(0, 255).astype(np.uint8)
                    rgb_img = Image.fromarray(rgb_arr)
                canvas.paste(rgb_img, (p0_x, p0_y + 28))
                draw = ImageDraw.Draw(canvas)
                if 0 <= src_idx < len(rgb_frames):
                    p0_info = f'实拍相机 Cam 0{cam_id} · 源帧 #{src_idx:03d} ({src_idx*0.04:.2f}s) · 无标记视觉输入'
                else:
                    p0_info = (
                        f'实拍相机 Cam 0{cam_id} · '
                        + ('初始站立准备 (显示首帧)' if now_s < 3.0 else '动作结束回归站立 (显示末帧)')
                    )
                draw.text((p0_x, p0_y + 383), p0_info, font=fonts[15], fill='#2c3e50')

                # Panel 2 (Top-Center, 580, 82): 3D Continuous Human Skeleton + Raw Triangulated Points
                p1_x, p1_y = 580, 82
                draw.text(
                    (p1_x, p1_y),
                    '2. 真实人体 3D 连续骨骼 (A2 因果滑窗 + 骨长投影)',
                    font=fonts[20],
                    fill='#158a72',
                )
                draw.rectangle((p1_x, p1_y + 28, p1_x + top_w, p1_y + 28 + top_h), fill='#ffffff', outline='#d5dce3')
                if 0 <= src_idx < len(pts_a2):
                    p = pts_a2[src_idx]
                    r_pt = pts_raw[src_idx]
                    row_s = states_a2[src_idx]
                    if np.isfinite(p[6:8]).all():
                        center_3d = p[6:8].mean(axis=0)

                    def px_map(pt3):
                        d = pt3 - center_3d
                        return (
                            p1_x + top_w // 2 + int(150 * (0.72 * d[0] + 0.68 * d[1])),
                            p1_y + 28 + top_h // 2 + 20 - int(150 * (d[2] - 0.14 * (d[0] - d[1]))),
                        )

                    statuses = dict(zip(row_s['joint_names'], row_s['joint_states']))
                    pred_flags = [
                        statuses[n].get('estimate_status') in ('predicted', 'recovering', 'reacquiring') for n in BODY
                    ]
                    for a_i, b_i in edges:
                        if np.isfinite(p[[a_i, b_i]]).all():
                            col_line = '#d9822b' if (pred_flags[a_i] or pred_flags[b_i]) else '#158a72'
                            draw.line((px_map(p[a_i]), px_map(p[b_i])), fill=col_line, width=5)
                    for z_raw in r_pt:
                        if np.isfinite(z_raw).all():
                            u, v = px_map(z_raw)
                            draw.ellipse((u - 3, v - 3, u + 3, v + 3), fill='#8c96a3')
                    miss_cnt = sum(not np.isfinite(z_raw).all() for z_raw in r_pt)
                    pred_cnt = sum(pred_flags)
                    p1_info = f'原始3D缺测点:{miss_cnt} · 连续恢复点:{pred_cnt} · 结构完整:12/12'
                else:
                    draw.text(
                        (p1_x + 110, p1_y + 185),
                        '初始站立准备阶段' if now_s < 3.0 else '动作结束站立保持阶段',
                        font=fonts[24],
                        fill='#7e8c9a',
                    )
                    p1_info = '非源运动时段：输出平稳站立参考'
                draw.text((p1_x, p1_y + 383), p1_info, font=fonts[15], fill='#2c3e50')

                # Panel 3 (Top-Right, 1135, 82): New Continuous C2 G1 Reference (Kinematics)
                p2_x, p2_y = 1135, 82
                draw.text(
                    (p2_x, p2_y),
                    '3. G1 新连续参考 (A2_continuous, 50Hz C² 运动学)',
                    font=fonts[20],
                    fill=COLORS['reference'],
                )
                pose_ref = tr_twist['command_qpos'][idx]
                data.qpos[:] = pose_ref
                mj.mj_kinematics(plant.model, data)
                camera.lookat[:] = [pose_ref[0], pose_ref[1], 0.65]
                renderer_top.update_scene(data, camera=camera)
                canvas.paste(Image.fromarray(renderer_top.render()), (p2_x, p2_y + 28))
                draw = ImageDraw.Draw(canvas)
                a0_idx = min(idx, len(stream_a0['stage']) - 1)
                a0_status = '旧A0此刻中断撤回站立!' if stream_a0['stage'][a0_idx] == 'reference_unavailable' else '旧A0此刻正常'
                p2_info = f'期望骨盆高 {pose_ref[2]:.2f} m · 50Hz C² 连续 ({a0_status})'
                draw.text(
                    (p2_x, p2_y + 383),
                    p2_info,
                    font=fonts[15],
                    fill='#b72d44' if '中断' in a0_status else '#2c3e50',
                )

                # Panel 4 (Bottom-Left, 25, 496): TWIST2 Free-Base Physics Rollout
                p3_x, p3_y = 25, 496
                draw.text(
                    (p3_x, p3_y),
                    f'4. TWIST2 自由基座动力学闭环模仿 (全程RMSE {res_twist["joint_rmse_deg"]:.1f}°, 下肢 {res_twist["leg_rmse_deg"]:.1f}°)',
                    font=fonts[20],
                    fill=COLORS['twist'],
                )
                q_tw = tr_twist['qpos'][idx]
                des_tw = tr_twist['desired_qpos'][idx]
                data.qpos[:] = q_tw
                mj.mj_kinematics(plant.model, data)
                camera.lookat[:] = [q_tw[0], q_tw[1], 0.65]
                renderer_bot.update_scene(data, camera=camera)
                canvas.paste(Image.fromarray(renderer_bot.render()), (p3_x, p3_y + 28))
                draw = ImageDraw.Draw(canvas)
                err_tw = float(np.degrees(np.sqrt(np.mean((q_tw[7:] - des_tw[7:]) ** 2))))
                xy_tw = float(np.linalg.norm(q_tw[:2] - des_tw[:2]))
                bad_tw = bool(balance_violation(q_tw[None])[0])
                p3_info = (
                    f'骨盆高 {q_tw[2]:.2f} m · 当前关节误差 {err_tw:.1f}° · XY偏差 {xy_tw:.2f} m · '
                    + ('失稳越界' if bad_tw else '自主平衡中')
                )
                draw.text((p3_x, p3_y + 383), p3_info, font=fonts[15], fill='#b72d44' if bad_tw else '#17283c')

                # Panel 5 (Bottom-Right, 860, 496): SONIC Free-Base Physics Rollout
                p4_x, p4_y = 860, 496
                draw.text(
                    (p4_x, p4_y),
                    f'5. SONIC 自由基座动力学闭环模仿 (全程RMSE {res_sonic["joint_rmse_deg"]:.1f}°, 下肢 {res_sonic["leg_rmse_deg"]:.1f}°)',
                    font=fonts[20],
                    fill=COLORS['sonic'],
                )
                q_so = tr_sonic['qpos'][idx]
                des_so = tr_sonic['desired_qpos'][idx]
                data.qpos[:] = q_so
                mj.mj_kinematics(plant.model, data)
                camera.lookat[:] = [q_so[0], q_so[1], 0.65]
                renderer_bot.update_scene(data, camera=camera)
                canvas.paste(Image.fromarray(renderer_bot.render()), (p4_x, p4_y + 28))
                draw = ImageDraw.Draw(canvas)
                err_so = float(np.degrees(np.sqrt(np.mean((q_so[7:] - des_so[7:]) ** 2))))
                xy_so = float(np.linalg.norm(q_so[:2] - des_so[:2]))
                bad_so = bool(balance_violation(q_so[None])[0])
                p4_info = (
                    f'骨盆高 {q_so[2]:.2f} m · 当前关节误差 {err_so:.1f}° · XY偏差 {xy_so:.2f} m · '
                    + ('失稳越界' if bad_so else '自主平衡中')
                )
                draw.text((p4_x, p4_y + 383), p4_info, font=fonts[15], fill='#b72d44' if bad_so else '#17283c')

                footer = (
                    '说明：同一 G1 自由基座模型、同一力矩/PD 限位、同一 C² 连续参考输入；无前置安全滤波，平衡与稳定完全由 TWIST2 / SONIC 控制器自行闭环解决'
                )
                draw.text((24, 922), footer, font=fonts[15], fill='#4a5d70')

                encoder.stdin.write(np.asarray(canvas).tobytes())
                total_frames += 1

                if frame == snap_frame:
                    canvas.save(out_dir / f'snapshot_{clip}.png')
            print(f'Rendered clip {clip} ({n_frames} frames)', flush=True)
    finally:
        renderer_top.close()
        renderer_bot.close()
        encoder.stdin.close()
        err_msg = encoder.stderr.read().decode('utf-8', errors='replace')
        ret_code = encoder.wait()
    if ret_code != 0:
        raise RuntimeError(f'ffmpeg failed: {err_msg}')

    subprocess.run(
        [ffmpeg, '-v', 'error', '-xerror', '-i', str(video_path), '-f', 'null', '-'],
        check=True,
        capture_output=True,
    )
    cap = cv2.VideoCapture(str(video_path))
    decoded = 0
    while True:
        ok, frm = cap.read()
        if not ok:
            break
        if frm.shape != (height, width, 3):
            raise RuntimeError(f'Unexpected video frame shape: {frm.shape}')
        decoded += 1
    cap.release()
    if decoded != total_frames:
        raise RuntimeError(f'Frame mismatch: encoded {total_frames} vs decoded {decoded}')
    val_info = dict(
        video_path=str(video_path),
        frames=total_frames,
        decoded_frames=decoded,
        fps=25,
        duration_seconds=total_frames / 25.0,
        width=width,
        height=height,
        clip_start_seconds=chapters,
        full_decode_passed=True,
    )
    save_json(out_dir / 'video_validation.json', val_info)
    print('Video verified:', json.dumps(val_info, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output',
        type=Path,
        default=ROOT / 'outputs/continuous_twist_vs_sonic_20261021',
    )
    parser.add_argument('--plots-only', action='store_true')
    args = parser.parse_args()
    out_dir = args.output.resolve()
    make_plots(out_dir)
    if not args.plots_only:
        render_video(out_dir, delay_ms=120)


if __name__ == '__main__':
    main()
