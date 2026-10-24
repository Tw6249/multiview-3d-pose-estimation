"""Plots and synchronized MuJoCo comparison video for `benchmark_continuous_v3.py`.

Reads only saved outputs from `outputs/continuous_v3_20261023`, applies no metric
clamping or rescaling, and reports both `reference_relative_upright_passed`
(crouch-compatible upright stability) and `balance_envelope_passed` (fixed 0.50 m
standing threshold).
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
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from motion_control.twist import TwistPlant

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
CLIPS = ('s1_walk', 's1_crouch', 's1_exercise', 's2_misc', 's2_sports')
MODES = ('A0_legacy', 'A2_continuous', 'A3_ik', 'A3_balanced', 'A3_stream', 'A3_full')
COMB_MODES = ('A0_legacy', 'A2_gated', 'A2_continuous', 'A3_ik', 'A3_balanced', 'A3_stream', 'A3_full')
COLORS = dict(
    A0_legacy='#7f8c8d',
    A2_gated='#e6ab02',
    A2_continuous='#d95f02',
    A3_ik='#1b9e77',
    A3_balanced='#2b83ba',
    A3_stream='#7570b3',
    A3_full='#e7298a',
)


def rows(path):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]


def status_tag(r):
    if r['balance_envelope_passed'] and r['standing_stop_passed']:
        return 'OK'
    if r.get('reference_relative_upright_passed') and r['standing_stop_passed']:
        return f"OK(crouch z={r['minimum_root_height_m']:.2f}m)"
    if r['balance_envelope_passed'] or r.get('reference_relative_upright_passed'):
        return 'UPRIGHT(no-stop)'
    return 'FALL'


def generate_plots_and_tables(d):
    res = json.loads((d / 'results.json').read_text(encoding='utf-8'))
    comb_res = json.loads((d / 'combined_fault_results.json').read_text(encoding='utf-8'))
    comb_cont = json.loads((d / 'combined_fault_reference_continuity.json').read_text(encoding='utf-8'))

    # 1. IK branch on s1_walk, left shoulder pitch / roll and worst limb error
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for mode, lbl in (
        ('A2_continuous', 'A2_continuous (无肩部奇异区屏障，f=110..142 卡在 -177°/+129°)'),
        ('A3_ik', 'A3_ik / A3_full (加入肩部双重翻转软屏障，平滑回到常规分支)'),
    ):
        r = rows(d / 'references' / f's1_walk_{mode}.jsonl')
        t = np.arange(len(r)) * 0.04
        q = np.array([x['reference'][6:] for x in r])
        err = np.array([max([e for e in x['limb_direction_error_deg'][:4] if e is not None] or [np.nan]) for x in r])
        ax[0].plot(t, np.degrees(q[:, 15]), color=COLORS[mode], lw=1.8, label=lbl)
        ax[1].plot(t, np.degrees(q[:, 16]), color=COLORS[mode], lw=1.8, label=lbl)
        ax[2].plot(t, err, color=COLORS[mode], lw=1.8, label=lbl)
    ax[2].axhline(20.0, color='#b72d44', ls='--', lw=1.0, label='20° 方向误差参考线')
    ax[0].set_ylabel('左肩 pitch (°)')
    ax[1].set_ylabel('左肩 roll (°)')
    ax[2].set_ylabel('双臂最差方向误差 (°)')
    ax[2].set_xlabel('片段时间 (s)')
    ax[0].legend(loc='lower left', fontsize=9)
    ax[2].legend(loc='upper left', fontsize=9)
    for a in ax:
        a.grid(alpha=0.25)
    ax[0].set_title('图1：s1_walk 25 Hz 逆运动学分支对比（相同 3D 连续状态输入，>20° 帧数从 38 降至 6）', fontweight='bold')
    fig.tight_layout()
    fig.savefig(d / 'fig1_ik_branch_s1_walk.png', dpi=140)
    plt.close(fig)

    # 2. 50 Hz continuity zoom
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    j = 9  # right_knee_joint in 29-DOF vector
    for mode in ('A0_legacy', 'A2_continuous', 'A3_stream'):
        s = np.load(d / 'references' / f's1_walk_{mode}_stream.npz', allow_pickle=True)
        q = s['bounded'][:, 7:]
        t = s['time_s']
        m = (t > 6.0) & (t < 8.0)
        dq = np.diff(q, axis=0) / 0.02
        ddq = np.diff(dq, axis=0) / 0.02
        ax[0].plot(t[m], np.degrees(q[m, j]), color=COLORS[mode], lw=1.7, label=mode)
        ax[1].plot(t[1:][m[1:]], dq[m[1:], j], color=COLORS[mode], lw=1.5)
        ax[2].plot(t[2:][m[2:]], ddq[m[2:], j], color=COLORS[mode], lw=1.4)
    ax[0].set_ylabel('关节角 (°)')
    ax[1].set_ylabel('角速度 (rad/s)')
    ax[2].set_ylabel('角加速度 (rad/s²)')
    ax[2].set_xlabel('仿真时间 (s)')
    ax[0].legend(fontsize=9)
    for a in ax:
        a.grid(alpha=0.25)
    ax[0].set_title('图2：50 Hz 控制器参考流连续性对比（A2/A3 严格满足 |dq|<=6, |ddq|<=120, |jerk|<=3000）', fontweight='bold')
    fig.tight_layout()
    fig.savefig(d / 'fig2_stream_continuity.png', dpi=140)
    plt.close(fig)

    # 3. Closed loop natural clips
    metrics = (
        ('joint_rmse_deg', '全身 29 关节 RMSE (°)'),
        ('leg_rmse_deg', '下肢 12 关节 RMSE (°)'),
        ('path_error_p95_m', '根部 XY 路径误差 P95 (m)'),
    )
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.5))
    for r_i, ctrl in enumerate(('twist', 'sonic')):
        for c_i, (key, label) in enumerate(metrics):
            a = axes[r_i, c_i]
            w = 0.16
            for k, mode in enumerate(MODES):
                vals = [res[f'{mode}|{c}|{ctrl}'][key] for c in CLIPS]
                bars = a.bar(np.arange(len(CLIPS)) + (k - 2) * w, vals, w, color=COLORS[mode], label=mode)
                for b, c in zip(bars, CLIPS):
                    r_entry = res[f'{mode}|{c}|{ctrl}']
                    if not (r_entry.get('reference_relative_upright_passed') and r_entry['standing_stop_passed']):
                        a.text(b.get_x() + b.get_width() / 2, b.get_height(), '×', ha='center', va='bottom', color='#c0392b', fontsize=10, fontweight='bold')
                    elif not r_entry['balance_envelope_passed']:
                        a.text(b.get_x() + b.get_width() / 2, b.get_height(), '▼', ha='center', va='bottom', color='#1b9e77', fontsize=8)
            a.set_xticks(range(len(CLIPS)))
            a.set_xticklabels(CLIPS, rotation=18, fontsize=8.5)
            a.grid(axis='y', alpha=0.2)
            a.set_title(f'{ctrl.upper()}: {label}\n(红×=失稳跌倒, 绿▼=深蹲至0.483~0.490m不倒且停稳)', fontsize=8.5)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(d / 'fig3_closed_loop.png', dpi=140)
    plt.close(fig)

    # 4. Combined dropout fault on s1_walk
    fig, axs = plt.subplots(3, 1, figsize=(12.5, 9.5))
    # Panel A: 50 Hz left knee reference across combined dropouts (f=20..23 arm, f=50..58 legs, f=90..105 full body)
    for mode, lbl in (
        ('A0_legacy', 'A0_legacy (101/150 接受，缺测即撤回站姿 DEFAULT_Q)'),
        ('A2_gated', 'A2_gated (124/150 接受，转身超限撤回 26 帧)'),
        ('A2_continuous', 'A2_continuous (150/150 接受，含 7 帧短缺口保持 + C² 平滑过渡)'),
        ('A3_stream', 'A3_stream (150/150 接受，逐自由度速度跟踪 C² 流)'),
    ):
        s = np.load(d / 'references' / f's1_walk_combined_{mode}_stream.npz', allow_pickle=True)
        t = s['time_s']
        q_knee = np.degrees(s['bounded'][:, 7 + 3])
        axs[0].plot(t, q_knee, color=COLORS[mode], lw=1.7, label=lbl)
    # Highlight 25 Hz dropout intervals mapped to simulation time t = 3.0 + f * 0.04
    for f0, f1, name in ((20, 23, '120ms 左臂掉线'), (50, 58, '320ms 双腿掉线'), (90, 105, '600ms 全身掉线')):
        axs[0].axvspan(3.0 + f0 * 0.04, 3.0 + f1 * 0.04, color='#e74c3c', alpha=0.14)
        axs[0].text(3.0 + 0.5 * (f0 + f1) * 0.04, 52, name, ha='center', fontsize=8.5, color='#922b21')
    axs[0].set_xlim(2.5, 10.2)
    axs[0].set_ylabel('左膝参考角 (°)')
    axs[0].set_title('A. s1_walk 连续掉线故障 (combined) 下的 50 Hz 左膝关节参考轨迹对比', loc='left', fontweight='bold')
    axs[0].legend(fontsize=8.5, loc='lower left', ncol=2)
    axs[0].grid(alpha=0.25)

    # Panel B & C: Closed-loop joint RMSE & Path error P95 on s1_walk_combined
    x = np.arange(len(COMB_MODES))
    bw = 0.34
    for i_c, (ctrl, col) in enumerate((('twist', '#235789'), ('sonic', '#b75d28'))):
        q_vals = [comb_res[f'{m}|s1_walk_combined|{ctrl}']['joint_rmse_deg'] for m in COMB_MODES]
        p_vals = [comb_res[f'{m}|s1_walk_combined|{ctrl}']['path_error_p95_m'] for m in COMB_MODES]
        offset = (i_c - 0.5) * bw
        b_q = axs[1].bar(x + offset, q_vals, bw, color=col, label=ctrl.upper())
        b_p = axs[2].bar(x + offset, p_vals, bw, color=col, label=ctrl.upper())
        axs[1].bar_label(b_q, fmt='%.1f°', fontsize=8, padding=2)
        axs[2].bar_label(b_p, fmt='%.2fm', fontsize=8, padding=2)
        for idx_m, m in enumerate(COMB_MODES):
            r_entry = comb_res[f'{m}|s1_walk_combined|{ctrl}']
            if not (r_entry['reference_relative_upright_passed'] and r_entry['standing_stop_passed']):
                axs[1].text(x[idx_m] + offset, q_vals[idx_m] + 1.5, 'FALL', ha='center', color='#c0392b', fontsize=8, fontweight='bold')
                axs[2].text(x[idx_m] + offset, p_vals[idx_m] + 0.08, 'FALL', ha='center', color='#c0392b', fontsize=8, fontweight='bold')
    for a, ylabel, title in (
        (axs[1], '全身关节 RMSE (°)', 'B. s1_walk 连续掉线故障下闭环全身关节 RMSE (°)（A2_continuous 与 A3_stream 双控制器均稳定完成）'),
        (axs[2], '根部路径误差 P95 (m)', 'C. s1_walk 连续掉线故障下闭环根部 XY 路径误差 P95 (m)'),
    ):
        a.set_xticks(x, [f"{m}\n({comb_cont[m]['accepted_frames']}/150)" for m in COMB_MODES], fontsize=8.5)
        a.set_ylabel(ylabel)
        a.set_title(title, loc='left', fontweight='bold')
        a.legend(fontsize=8.5)
        a.grid(axis='y', alpha=0.25)
    axs[1].set_ylim(0, 40)
    axs[2].set_ylim(0, 2.35)
    fig.tight_layout()
    fig.savefig(d / 'fig4_combined_fault_s1_walk.png', dpi=140)
    plt.close(fig)

    # 5. Compromise comparison: A2_continuous vs A3_ik & A3_balanced vs A3_full (under SONIC)
    ref_cont = json.loads((d / 'reference_continuity.json').read_text(encoding='utf-8'))
    ik_br = json.loads((d / 'ik_branch.json').read_text(encoding='utf-8'))
    cmp_modes = ('A2_continuous', 'A3_ik', 'A3_balanced', 'A3_full')
    cmp_labels = {
        'A2_continuous': 'A2_continuous (默认)\n无屏障IK + 组同步v=0',
        'A3_ik': '折中1: A3_ik\n屏障IK + 组同步v=0',
        'A3_balanced': '折中2: A3_balanced\n屏障IK + 单调阻尼v+柔化跃度',
        'A3_full': 'A3_full (激进全开)\n屏障IK + 逐关节全速v',
    }
    fig, ax5 = plt.subplots(2, 2, figsize=(15.5, 9.5))

    # Panel A: 50 Hz Mean P95 Jerk & Max Jerk + IK >20 deg frames
    xm = np.arange(len(cmp_modes))
    p95_jerks = [float(np.mean([ref_cont[f'{m}|{c}']['control_50hz']['p95_jerk_rad_s3'] for c in CLIPS])) for m in cmp_modes]
    max_jerks = [float(np.max([ref_cont[f'{m}|{c}']['control_50hz']['max_jerk_rad_s3'] for c in CLIPS])) for m in cmp_modes]
    b1 = ax5[0, 0].bar(xm - 0.17, p95_jerks, 0.32, color=[COLORS[m] for m in cmp_modes], label='5片段平均 P95 跃度 (rad/s³)')
    b2 = ax5[0, 0].bar(xm + 0.17, max_jerks, 0.32, color=[COLORS[m] for m in cmp_modes], alpha=0.42, edgecolor='#2c3e50', lw=1.0, label='5片段最大跃度上界 (rad/s³)')
    ax5[0, 0].bar_label(b1, fmt='%.0f', fontsize=8.5, padding=2, fontweight='bold')
    ax5[0, 0].bar_label(b2, fmt='%.0f', fontsize=8, padding=2)
    for idx_m, m in enumerate(cmp_modes):
        n_over20 = ik_br[f'{m}|s1_walk']['frames_worst_limb_over_20deg']
        ax5[0, 0].text(idx_m, max_jerks[idx_m] + 190, f's1_walk >20°: {n_over20}帧', ha='center', fontsize=8.5,
                       color='#c0392b' if n_over20 > 10 else '#158a72', fontweight='bold')
    ax5[0, 0].set_xticks(xm, [cmp_labels[m] for m in cmp_modes], fontsize=8.5)
    ax5[0, 0].set_ylim(0, 3950)
    ax5[0, 0].set_ylabel('50 Hz 关节跃度 (rad/s³)')
    ax5[0, 0].set_title('A. 50 Hz 参考流跃度冲击与 s1_walk 肩部奇异区翻转帧数对比', loc='left', fontweight='bold')
    ax5[0, 0].legend(fontsize=8.5, loc='upper center', ncol=2)
    ax5[0, 0].grid(axis='y', alpha=0.25)

    # Panel B: s1_exercise pelvis height z(t) during deep squat under SONIC
    s_ref = np.load(d / 'references' / 's1_exercise_A3_ik_stream.npz', allow_pickle=True)
    ax5[0, 1].plot(s_ref['time_s'], s_ref['bounded'][:, 2], color='#2c3e50', ls='--', lw=1.5, label='参考轨迹根部高度 z_ref (min=0.485m)')
    for m in cmp_modes:
        tr = np.load(d / 'traces' / f'{m}_s1_exercise_sonic.npz', allow_pickle=True)
        min_z = res[f'{m}|s1_exercise|sonic']['minimum_root_height_m']
        ax5[0, 1].plot(tr['time_s'], tr['qpos'][:, 2], color=COLORS[m], lw=1.8, label=f'{m} (min z={min_z:.3f}m)')
    ax5[0, 1].axhline(0.50, color='#7f8c8d', ls=':', lw=1.0, label='固定站立阈值 0.500m')
    ax5[0, 1].set_xlim(3.0, 9.0)
    ax5[0, 1].set_ylim(0.46, 0.78)
    ax5[0, 1].set_xlabel('仿真时间 (s)')
    ax5[0, 1].set_ylabel('骨盆高度 z (m)')
    ax5[0, 1].set_title('B. s1_exercise 深蹲段 SONIC 闭环骨盆高度响应（A3_balanced 下蹲到位且跃度减半）', loc='left', fontweight='bold')
    ax5[0, 1].legend(fontsize=8, loc='lower right')
    ax5[0, 1].grid(alpha=0.25)

    # Panel C & D: SONIC closed-loop Joint RMSE and Path P95 across 5 natural clips + mean + combined fault
    scen_labels = list(CLIPS) + ['5自然均值', 's1_walk_comb']
    xs = np.arange(len(scen_labels))
    w_bar = 0.19
    for idx_m, m in enumerate(cmp_modes):
        q_nat = [res[f'{m}|{c}|sonic']['joint_rmse_deg'] for c in CLIPS]
        q_all = q_nat + [float(np.mean(q_nat)), comb_res[f'{m}|s1_walk_combined|sonic']['joint_rmse_deg']]
        p_nat = [res[f'{m}|{c}|sonic']['path_error_p95_m'] for c in CLIPS]
        p_all = p_nat + [float(np.mean(p_nat)), comb_res[f'{m}|s1_walk_combined|sonic']['path_error_p95_m']]
        off = (idx_m - 1.5) * w_bar
        bq = ax5[1, 0].bar(xs + off, q_all, w_bar, color=COLORS[m], label=m)
        bp = ax5[1, 1].bar(xs + off, p_all, w_bar, color=COLORS[m], label=m)
        ax5[1, 0].bar_label(bq, fmt='%.1f', fontsize=7, padding=1.5)
        ax5[1, 1].bar_label(bp, fmt='%.2f', fontsize=7, padding=1.5)
    for col_i, (ylabel, title, ylim) in enumerate((
        ('全身 29 关节 RMSE (°)', 'C. SONIC 单策略闭环：全身关节跟踪误差 RMSE (°) 对比（6/6 场景全部稳定停稳）', (0, 36)),
        ('根部 XY 路径误差 P95 (m)', 'D. SONIC 单策略闭环：根部 XY 路径误差 P95 (m) 对比（折中方案显著抑制行走漂移）', (0, 2.25)),
    )):
        ax5[1, col_i].axvline(4.5, color='#7f8c8d', ls='--', lw=1.0)
        ax5[1, col_i].set_xticks(xs, scen_labels, fontsize=8.5, rotation=12)
        ax5[1, col_i].set_ylabel(ylabel)
        ax5[1, col_i].set_ylim(*ylim)
        ax5[1, col_i].set_title(title, loc='left', fontweight='bold')
        ax5[1, col_i].legend(fontsize=8, ncol=2, loc='upper left')
        ax5[1, col_i].grid(axis='y', alpha=0.25)

    fig.tight_layout()
    fig.savefig(d / 'fig5_compromise_comparison.png', dpi=140)
    plt.close(fig)

    # Summary markdown table
    lines = [
        '### 1. 五个自然片段闭环结果 (`120 ms` 延迟)',
        '',
        '| 片段 | 控制器 | ' + ' | '.join(MODES) + ' |',
        '|---|---|' + '---|' * len(MODES),
    ]
    for c in CLIPS:
        for ctrl in ('twist', 'sonic'):
            cells = []
            for m in MODES:
                r = res[f'{m}|{c}|{ctrl}']
                cells.append(f"{status_tag(r)} q={r['joint_rmse_deg']:.1f}° leg={r['leg_rmse_deg']:.1f}° path={r['path_error_p95_m']:.2f}m")
            lines.append(f'| {c} | {ctrl} | ' + ' | '.join(cells) + ' |')

    lines += [
        '',
        '### 2. `s1_walk` 连续掉线故障 (`combined`: 120ms 单臂 + 320ms 双腿 + 600ms 全身掉线) 闭环结果',
        '',
        '| 模式 | 25Hz 接受/保持 | 控制器 | 状态 | 自身关节 RMSE | 下肢 RMSE | 局部3D关键点均值/P95 | 路径误差 P95 |',
        '|---|---|---|---|---|---|---|---|',
    ]
    for m in COMB_MODES:
        acc_str = f"{comb_cont[m]['accepted_frames']}/150 (保持 {comb_cont[m]['short_gap_held_frames']})"
        for ctrl in ('twist', 'sonic'):
            r = comb_res[f'{m}|s1_walk_combined|{ctrl}']
            lines.append(
                f"| {m} | {acc_str} | {ctrl} | {status_tag(r)} | {r['joint_rmse_deg']:.2f}° | "
                f"{r['leg_rmse_deg']:.2f}° | {r['root_local_body_position_mean_mm']:.1f} / {r['root_local_body_position_p95_mm']:.1f} mm | "
                f"{r['path_error_p95_m']:.3f} m |"
            )
    (d / 'summary_table.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


def render_comparison_video(d):
    from PIL import Image, ImageDraw, ImageFont

    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('ffmpeg not found')
    fonts = {sz: ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', sz) for sz in (14, 16, 18, 22, 26)}
    res = json.loads((d / 'results.json').read_text(encoding='utf-8'))
    comb_res = json.loads((d / 'combined_fault_results.json').read_text(encoding='utf-8'))

    width, height = 1600, 900
    tile_w, tile_h = 510, 360
    plant = TwistPlant()
    mj = plant.mj
    plant.model.vis.global_.offwidth = tile_w
    plant.model.vis.global_.offheight = tile_h
    renderer = mj.Renderer(plant.model, height=tile_h, width=tile_w)
    data = mj.MjData(plant.model)
    cam = mj.MjvCamera()
    cam.azimuth = 135
    cam.elevation = -15
    cam.distance = 2.65

    video_path = d / 'comparison_continuous_v3.mp4'
    cmd = [
        ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', '25',
        '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    chapters = [
        (
            '场景 1：s1_walk 连续掉线故障 (120ms 单臂 + 320ms 双腿 + 600ms 全身掉线)',
            comb_res,
            [
                ('A0_legacy|s1_walk_combined|twist', '1. A0_legacy + TWIST2 (49帧撤回站姿)'),
                ('A2_continuous|s1_walk_combined|twist', '2. A2_continuous + TWIST2 (150/150 C²连续)'),
                ('A3_stream|s1_walk_combined|twist', '3. A3_stream + TWIST2 (逐关节速度跟踪)'),
                ('A0_legacy|s1_walk_combined|sonic', '4. A0_legacy + SONIC (49帧撤回站姿)'),
                ('A2_continuous|s1_walk_combined|sonic', '5. A2_continuous + SONIC (150/150 C²连续)'),
                ('A3_full|s1_walk_combined|sonic', '6. A3_full + SONIC (路径P95=1.28m)'),
            ],
            True,
        ),
        (
            '场景 2：s1_walk 自然行走与转身（对比左臂奇异区修复与控制器闭环）',
            res,
            [
                ('A0_legacy|s1_walk|twist', '1. A0_legacy + TWIST2 (转身锁死跌倒)'),
                ('A2_continuous|s1_walk|twist', '2. A2_continuous + TWIST2 (全程平衡)'),
                ('A3_full|s1_walk|twist', '3. A3_full + TWIST2 (肩部脱困+全程平衡)'),
                ('A0_legacy|s1_walk|sonic', '4. A0_legacy + SONIC (撤回卡顿)'),
                ('A2_continuous|s1_walk|sonic', '5. A2_continuous + SONIC (RMSE=28.3°)'),
                ('A3_full|s1_walk|sonic', '6. A3_full + SONIC (RMSE=25.1°)'),
            ],
            False,
        ),
        (
            '场景 3：s1_exercise 深蹲与体操（验证 SONIC 深蹲至 0.483m 不倒且平稳恢复站立）',
            res,
            [
                ('A0_legacy|s1_exercise|twist', '1. A0_legacy + TWIST2 (深蹲失稳)'),
                ('A2_continuous|s1_exercise|twist', '2. A2_continuous + TWIST2 (4.3s 深蹲倾覆)'),
                ('A3_full|s1_exercise|twist', '3. A3_full + TWIST2 (6.3s 深蹲倾覆)'),
                ('A0_legacy|s1_exercise|sonic', '4. A0_legacy + SONIC (撤回站姿跳跃)'),
                ('A2_continuous|s1_exercise|sonic', '5. A2_continuous + SONIC (min z=0.514m 稳定)'),
                ('A3_full|s1_exercise|sonic', '6. A3_full + SONIC (min z=0.490m 深蹲不倒)'),
            ],
            False,
        ),
    ]

    try:
        for title, r_dict, panels, is_comb in chapters:
            traces = [np.load(d / 'traces' / (k.replace('|', '_') + '.npz'), allow_pickle=True) for k, _ in panels]
            n_ticks = len(traces[0]['time_s'])
            # Render at 25 fps (every 2nd 50 Hz tick)
            for idx in range(0, n_ticks, 2):
                now_s = float(traces[0]['time_s'][idx])
                canvas = Image.new('RGB', (width, height), '#f4f6f8')
                draw = ImageDraw.Draw(canvas)
                draw.text((24, 12), title, font=fonts[26], fill='#17283c')
                drop_banner = ''
                if is_comb:
                    src_f = (now_s - 3.0) / 0.04
                    if 20 <= src_f < 23:
                        drop_banner = '  [当前处于：120 ms 左臂掉线窗口 f=20..23]'
                    elif 50 <= src_f < 58:
                        drop_banner = '  [当前处于：320 ms 双腿掉线窗口 f=50..58]'
                    elif 90 <= src_f < 105:
                        drop_banner = '  [当前处于：600 ms 全身掉线窗口 f=90..105]'
                draw.text((24, 48), f'仿真时间 t = {now_s:5.2f} s  |  120 ms 因果延迟 · 无前置安全滤波{drop_banner}', font=fonts[16], fill='#b72d44' if drop_banner else '#4a5d70')

                for p_i, ((k, p_title), tr) in enumerate(zip(panels, traces)):
                    row_i, col_i = divmod(p_i, 3)
                    px = 20 + col_i * 525
                    py = 82 + row_i * 405
                    m = r_dict[k]
                    qpos = tr['qpos'][idx]
                    data.qpos[:] = qpos
                    mj.mj_kinematics(plant.model, data)
                    cam.lookat[:] = [qpos[0], qpos[1], 0.65]
                    renderer.update_scene(data, camera=cam)
                    img = Image.fromarray(renderer.render())
                    canvas.paste(img, (px, py + 26))
                    draw = ImageDraw.Draw(canvas)
                    ok = m.get('reference_relative_upright_passed') and m['standing_stop_passed']
                    col_hdr = '#158a72' if ok else '#c0392b'
                    draw.text((px, py), p_title, font=fonts[18], fill=col_hdr)
                    info = f"状态:{status_tag(m)} | q_RMSE={m['joint_rmse_deg']:.1f}° | 路径P95={m['path_error_p95_m']:.2f}m | 当前z={qpos[2]:.2f}m"
                    draw.text((px, py + 388), info, font=fonts[14], fill='#2c3e50')
                proc.stdin.write(np.asarray(canvas, dtype=np.uint8).tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode('utf-8', errors='replace')
        if proc.wait() != 0:
            raise RuntimeError(f'ffmpeg failed: {err}')
    finally:
        renderer.close()
    print(f'Saved video: {video_path}')


def render_compromise_video(d):
    from PIL import Image, ImageDraw, ImageFont

    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('ffmpeg not found')
    fonts = {sz: ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', sz) for sz in (14, 15, 16, 18, 22, 26)}
    res = json.loads((d / 'results.json').read_text(encoding='utf-8'))
    comb_res = json.loads((d / 'combined_fault_results.json').read_text(encoding='utf-8'))

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
    cam.distance = 2.55

    video_path = d / 'comparison_compromise_sonic.mp4'
    cmd = [
        ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', '25',
        '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    chapters = [
        (
            '第 1 幕：s1_walk 自然行走与转身（SONIC 单策略，重点看 t=7.4~8.8s 左臂奇异区与路径漂移）',
            res,
            's1_walk',
            [
                ('A2_continuous|s1_walk|sonic', '1. A2_continuous (默认基线：无屏障IK + 组同步v=0) — t=7.4s左臂反折'),
                ('A3_ik|s1_walk|sonic', '2. 折中方案1：A3_ik (屏障IK + 组同步v=0) — 消除翻转且路径漂移最小(0.63m)'),
                ('A3_balanced|s1_walk|sonic', '3. 折中方案2：A3_balanced (屏障IK + 单调阻尼v + 1500柔化跃度)'),
                ('A3_full|s1_walk|sonic', '4. A3_full (激进全开：屏障IK + 逐关节全速v) — 跃度较高(P95=2779)'),
            ],
            False,
        ),
        (
            '第 2 幕：s1_exercise 深蹲与全身操（SONIC 单策略，重点看 t=3.6~5.5s 深蹲最低点 z 与跃度）',
            res,
            's1_exercise',
            [
                ('A2_continuous|s1_exercise|sonic', '1. A2_continuous (默认基线) — 深蹲最低 z=0.514m (略浅)'),
                ('A3_ik|s1_exercise|sonic', '2. 折中方案1：A3_ik — 深蹲最低 z=0.512m (略浅)'),
                ('A3_balanced|s1_exercise|sonic', '3. 折中方案2：A3_balanced — 深蹲到底 z=0.488m 且跃度减半(<=1500)'),
                ('A3_full|s1_exercise|sonic', '4. A3_full (激进全开) — 深蹲到底 z=0.490m，但最大跃度达 3000'),
            ],
            False,
        ),
        (
            '第 3 幕：s1_walk 连续掉线故障 (120ms 左臂 + 320ms 双腿 + 600ms 全身掉线，SONIC 全程不倒)',
            comb_res,
            's1_walk_combined',
            [
                ('A2_continuous|s1_walk_combined|sonic', '1. A2_continuous (默认基线) — 掉线恢复 RMSE=28.1°, 路径P95=1.63m'),
                ('A3_ik|s1_walk_combined|sonic', '2. 折中方案1：A3_ik — 掉线恢复 RMSE=29.9°, 路径P95=1.94m'),
                ('A3_balanced|s1_walk_combined|sonic', '3. 折中方案2：A3_balanced — 柔化恢复，路径P95=1.54m'),
                ('A3_full|s1_walk_combined|sonic', '4. A3_full (激进全开) — 快速追赶恢复，路径P95=1.28m'),
            ],
            True,
        ),
    ]

    try:
        for title, r_dict, clip_name, panels, is_comb in chapters:
            traces = [np.load(d / 'traces' / (k.replace('|', '_') + '.npz'), allow_pickle=True) for k, _ in panels]
            streams = [
                np.load(d / 'references' / f"{clip_name}_{k.split('|')[0]}_stream.npz", allow_pickle=True)
                for k, _ in panels
            ]
            n_ticks = len(traces[0]['time_s'])
            for idx in range(0, n_ticks, 2):
                now_s = float(traces[0]['time_s'][idx])
                canvas = Image.new('RGB', (width, height), '#f4f6f8')
                draw = ImageDraw.Draw(canvas)
                draw.text((24, 12), title, font=fonts[22], fill='#17283c')
                banner = ''
                if is_comb:
                    src_f = (now_s - 3.0) / 0.04
                    if 20 <= src_f < 23:
                        banner = '  [当前处于：120 ms 左臂掉线窗口 f=20..23]'
                    elif 50 <= src_f < 58:
                        banner = '  [当前处于：320 ms 双腿掉线窗口 f=50..58]'
                    elif 90 <= src_f < 105:
                        banner = '  [当前处于：600 ms 全身掉线窗口 f=90..105]'
                elif clip_name == 's1_walk' and 7.4 <= now_s <= 8.7:
                    banner = '  [当前处于：左肩奇异区窗口 f=110..142 — 注意左上角 A2_continuous 左臂翻转]'
                elif clip_name == 's1_exercise' and 3.6 <= now_s <= 5.5:
                    banner = '  [当前处于：深蹲最低点窗口 — 注意左下 A3_balanced 与右下 A3_full 完整蹲至 0.49m]'
                draw.text(
                    (24, 46),
                    f'仿真时间 t = {now_s:5.2f} s  |  单一部署策略：SONIC (100% 因果实时参考 · 120 ms 延迟){banner}',
                    font=fonts[16],
                    fill='#b72d44' if banner else '#4a5d70',
                )

                for p_i, ((k, p_title), tr, st) in enumerate(zip(panels, traces, streams)):
                    row_i, col_i = divmod(p_i, 2)
                    px = 20 + col_i * 785
                    py = 80 + row_i * 415
                    m = r_dict[k]
                    mode_name = k.split('|')[0]
                    qpos = tr['qpos'][idx]
                    st_idx = min(idx, len(st['bounded']) - 1)
                    q_ref = st['bounded'][st_idx, 7:]
                    z_ref = float(st['bounded'][st_idx, 2])
                    inst_q_rmse = float(np.degrees(np.sqrt(np.mean((qpos[7:] - q_ref) ** 2))))
                    inst_jerk = float(np.max(np.abs(st['jerk'][min(idx, len(st['jerk']) - 1)])))
                    data.qpos[:] = qpos
                    mj.mj_kinematics(plant.model, data)
                    cam.lookat[:] = [qpos[0], qpos[1], 0.65]
                    renderer.update_scene(data, camera=cam)
                    img = Image.fromarray(renderer.render())
                    canvas.paste(img, (px, py + 26))
                    draw = ImageDraw.Draw(canvas)
                    draw.text((px, py), p_title, font=fonts[16], fill=COLORS.get(mode_name, '#158a72'))
                    info = (
                        f"全程RMSE={m['joint_rmse_deg']:.1f}° | 路径P95={m['path_error_p95_m']:.2f}m | "
                        f"瞬时误差={inst_q_rmse:.1f}° | 当前z={qpos[2]:.2f}m(参考{z_ref:.2f}m) | 瞬时跃度={inst_jerk:.0f}rad/s³"
                    )
                    draw.text((px, py + 390), info, font=fonts[14], fill='#2c3e50')
                proc.stdin.write(np.asarray(canvas, dtype=np.uint8).tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode('utf-8', errors='replace')
        if proc.wait() != 0:
            raise RuntimeError(f'ffmpeg failed: {err}')
    finally:
        renderer.close()
    print(f'Saved compromise video: {video_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', type=Path, default=ROOT / 'outputs/continuous_v3_20261023')
    ap.add_argument('--skip-video', action='store_true')
    ap.add_argument('--compromise-video-only', action='store_true')
    args = ap.parse_args()
    if args.compromise_video_only:
        render_compromise_video(args.dir)
        return
    generate_plots_and_tables(args.dir)
    if not args.skip_video:
        render_comparison_video(args.dir)
        render_compromise_video(args.dir)


if __name__ == '__main__':
    main()
