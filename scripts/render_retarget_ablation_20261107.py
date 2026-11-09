"""Render saved ablations, no re-simulation or display-only pose modification.

1920x1080, source 30 Hz, five equal columns, fixed shared camera for all variants.
Top: two synchronized RGB cameras and input skeleton. Middle: kinematic references.
Bottom: actual MuJoCo closed-loop qpos, including failures/drift. No time warping.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retarget_ablation_20261107 import OUT, EXPERIMENT, configure_runtime, VARIANTS
from advanced_take_render_rgb_control import draw27
from advanced_take_render_video import Reader
from g1_imitation.core import adv_eval as ae
from g1_imitation.core.s6_mujoco_closed_loop import G1PhysicsPlant

W, H, PW, PH = 1920, 1080, 384, 300
FONT = {s: ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", s) for s in (16, 18, 22, 28)}
LABELS = {"baseline": "基线", "relations": "① 身体关系", "support": "② 骨盆 / 支撑", "temporal": "③ 时序 / 限速", "combined": "①②③ 组合"}
COLORS = [(85, 85, 85), (206, 115, 45), (80, 150, 70), (150, 85, 165), (30, 135, 210)]


def text(image, items):
    p = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(p)
    for xy, label, size, color in items:
        draw.text(xy, label, font=FONT[size], fill=tuple(reversed(color)))
    return cv2.cvtColor(np.asarray(p), cv2.COLOR_RGB2BGR)


def input_panel(points, paths, k, extent):
    image = np.full((PH, 640, 3), 249, np.uint8)
    # Same fixed oblique viewing direction throughout; source skeleton is translated for visibility only.
    p = points.copy()
    mid = np.nanmean(p[[11, 12]], axis=0)
    p[:, :2] -= mid[:2]
    az, el = np.radians(38), np.radians(22)
    xx = np.cos(az) * p[:, 0] - np.sin(az) * p[:, 1]
    yy = np.sin(az) * p[:, 0] + np.cos(az) * p[:, 1]
    uv = np.c_[175 + xx * 145, 270 - (np.cos(el) * p[:, 2] + np.sin(el) * yy) * 145]
    draw27(image, uv, th=2)
    x0, y0, ww, hh = 350, 40, 270, 235
    xmin, xmax, ymin, ymax = extent
    def pixel(x):
        return np.c_[x0 + (x[:, 0] - xmin) / (xmax - xmin) * ww,
                     y0 + hh - (x[:, 1] - ymin) / (ymax - ymin) * hh].astype(np.int32)
    cv2.rectangle(image, (x0, y0), (x0 + ww, y0 + hh), (210, 210, 210), 1)
    for i, path in enumerate(paths):
        pth = path[:k + 1]
        if len(pth) > 1:
            cv2.polylines(image, [pixel(pth)], False, COLORS[i], 2, cv2.LINE_AA)
    return text(image, [((x0, y0 + hh + 1), f"X: {xmin:.1f}～{xmax:.1f} m", 16, (95, 95, 95)),
                        ((x0 + 6, y0 + 5), f"Y: {ymin:.1f}～{ymax:.1f} m", 16, (95, 95, 95))])


def render(take, fps=30):
    cfg, assets = configure_runtime()
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    saved = {v: dict(np.load(OUT / take / f"{v}.npz")) for v in VARIANTS}
    z = np.load(ROOT / "outputs" / "advanced_take" / "s2s3" / "rgbonly_lag6" / f"{take}.npz")
    grid, points = z["grid"], z["smooth"]
    lag = int(json.loads((ROOT / "outputs" / "advanced_take" / "s2s3" / "rgbonly_lag6" / "metrics.json").read_text(encoding="utf-8"))["ref_lag_bvh_samples"])
    slot_times = (grid + lag) / 120
    s1 = np.load(ROOT / "outputs" / "advanced_take" / "s1" / f"{take}_kp27.npz")
    camera_ids = ["44310274", "44310281"]
    cams = json.loads((ROOT / "outputs" / "advanced_take" / "cameras_rgbonly.json").read_text(encoding="utf-8"))["cameras"]
    grid_check, fidx = ae.build_slot_grid({c: s1[f"{c}_pts_s"] for c in camera_ids}, int(grid[-1] + 1))
    if not np.array_equal(grid_check, grid):
        raise RuntimeError("RGB grid mismatch")
    readers = [Reader(ROOT / "data" / "advanced_take_raw" / f"{take}_Video_{c}.mp4") for c in camera_ids]
    plant = G1PhysicsPlant(mjcf_path=assets / "sonic_official_g1" / "scene_43dof.xml")
    model = plant.kin_model
    model.vis.global_.offwidth, model.vis.global_.offheight = max(model.vis.global_.offwidth, PW), max(model.vis.global_.offheight, PH)
    renderer = mujoco.Renderer(model, height=PH, width=PW)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.distance, camera.elevation, camera.azimuth = 3.8, -19, 48
    paths = []
    for v in VARIANTS:
        ticks = np.clip(np.searchsorted(saved[v]["t50"], slot_times), 0, len(saved[v]["t50"]) - 1)
        paths.append(saved[v]["qpos_phys"][ticks, :2])
    all_paths = np.concatenate(paths)
    xmin, ymin = all_paths.min(0) - .2
    xmax, ymax = all_paths.max(0) + .2
    extent = xmin, xmax, ymin, ymax
    dest = OUT / f"{take}_comparison.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest)], stdin=subprocess.PIPE)
    sample_indices = {round((len(grid) - 1) * x) for x in (.25, .5, .75)}
    stills = []
    for k, t in enumerate(slot_times):
        canvas = np.full((H, W, 3), 247, np.uint8)
        annotations = [((15, 8), f"{take} · 重定向改进对照 · 同一 RGB、时钟、G1 模型与 SONIC v1.1@0", 28, (55, 55, 55)),
                       ((15, 45), f"t = {t-slot_times[0]:.2f} s · 原速 1× · 原时刻比较（未按分支对齐动作）· 仅仿真", 18, (95, 95, 95))]
        for i, reader in enumerate(readers):
            f = int(fidx[camera_ids[i]][k])
            fr = reader.frame(f) if f >= 0 else None
            if fr is not None:
                tile = cv2.resize(fr, (398, 300), interpolation=cv2.INTER_AREA)
                # Preserve source aspect; centered inside equal-width panels.
                canvas[75:375, i * 640 + 121:i * 640 + 519] = tile
            annotations.append(((i * 640 + 12, 78), f"RGB {camera_ids[i]}", 18, (70, 70, 70)))
        canvas[75:375, 1280:1920] = input_panel(points[k], paths, k, extent)
        annotations.extend([((1290, 78), "视觉骨架（S3）", 18, (60, 60, 60)),
                            ((1630, 78), "实际根部 XY 轨迹", 18, (60, 60, 60))])
        base = saved["baseline"]
        bi = int(np.clip(np.searchsorted(base["t50"], t), 0, len(base["t50"]) - 1))
        camera.lookat[:] = [base["q_ref"][bi, 0], base["q_ref"][bi, 1], .67]
        for i, variant in enumerate(VARIANTS):
            rec = saved[variant]
            tick = int(np.clip(np.searchsorted(rec["t50"], t), 0, len(rec["t50"]) - 1))
            x = i * PW
            for y, key in ((410, "q_ref"), (745, "qpos_phys")):
                if t < rec["t50"][0]:
                    continue
                data.qpos[:] = rec[key][tick]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                canvas[y:y+PH, x:x+PW] = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            metrics = report["results"][take][variant]
            label = LABELS[variant]
            annotations.append(((x + 8, 382), f"{label} · 运动学参考", 18, COLORS[i]))
            annotations.append(((x + 8, 713), f"{label} · 物理跟踪", 18, COLORS[i]))
            annotations.append(((x + 8, 1048), f"臂 {metrics['arms_physics_direction_deg']:.1f}° / 腿 {metrics['legs_physics_direction_deg']:.1f}°", 16, COLORS[i]))
        canvas = text(canvas, annotations)
        ff.stdin.write(canvas.tobytes())
        if k in sample_indices:
            name = OUT / f"{take}_frame_{k:04d}.png"
            cv2.imencode(".png", canvas)[1].tofile(str(name))
            stills.append(name.name)
        if k % 100 == 0:
            print(take, f"render {k}/{len(grid)}", flush=True)
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    renderer.close()
    for reader in readers:
        reader.cap.release()
    print("Saved", dest, flush=True)


def gallery():
    r = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    columns = ("arms_reference_direction_deg", "arms_physics_direction_deg", "legs_physics_direction_deg", "support_foot_skating_m_s", "path_end_error_m")
    rows = []
    for v in VARIANTS:
        data = [d[v] for d in r["results"].values()]
        numbers = [np.mean([x[c] for x in data]) for c in columns]
        rows.append(f"<tr><td>{LABELS[v]}</td>" + "".join(f"<td>{n:.3f}</td>" for n in numbers) + f"<td>{sum(x['survived'] for x in data)}/5</td></tr>")
    videos = "".join(f'<section><h2>{take}</h2><video controls preload="metadata" src="{take}_comparison.mp4" poster="{take}_frame_{round((len(np.load(OUT/take/"baseline.npz")["q_slots"])-1)*.5):04d}.png"></video></section>' for take in TAKES if (OUT/f"{take}_comparison.mp4").exists())
    content = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>G1 重定向实验对照</title>
<style>body{{font:16px/1.7 system-ui;background:#f1f3f6;color:#253145;max-width:1400px;margin:30px auto;padding:0 22px}}section{{background:white;padding:22px;margin:22px 0;border-radius:14px}}video,img{{width:100%;border-radius:8px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;border-bottom:1px solid #ddd;text-align:right}}td:first-child,th:first-child{{text-align:left}}.note{{background:#fff0cd;padding:18px;border-radius:10px}}a{{color:#1759a5}}</style>
<h1>G1 重定向改进：五组同步对照</h1><p>2026-11-07 · 5 段真人数据 · 无标记 RGB 输入 · BVH 仅离线评分 · 当前结果仅来自仿真。</p>
<p class="note">请同时观察运动学参考（上排）与物理跟踪（下排）。各分支的模型、控制器、增益、仿真步长与预览设置相同。姿态误差改善不代表脚滑、路线或总体观感改善。视频按源时刻播放，没有按分支平移时间，也没有隐藏失败。所有结果未替换默认算法。</p>
<section><h2>五段平均值（越低越好）</h2><table><tr><th>方法</th><th>参考手臂 °</th><th>物理手臂 °</th><th>物理腿 °</th><th>脚滑 m/s</th><th>终点偏差 m</th><th>原存活判据</th></tr>{''.join(rows)}</table><p>方向误差在各自骨盆航向系内计算；航向/路线需另看。终点偏差为物理机器人相对自身参考；两者均不是完整模仿成功率。统一排除开始 1.2 秒的校准与启动区间（仅方向评分），不按分支移动时间。</p></section>
<section><h2>逐段指标</h2><img src="summary.png"><p><a href="report.json">完整 JSON 指标与实验参数</a> · <a href="diagnostics.json">额外轨迹/时序诊断</a></p></section>{videos}
<p>② 为骨盆高度软目标、降低不可观测骨盆俯仰先验权重与接触位置任务的联合实验；③ 为固定骨长的因果方向滤波＋关节速度/加速度约束。当前单项实验尚未拆分其内部因素，结论只适用于这些具体实现与参数。</p></html>'''
    (OUT / "index.html").write_text(content, encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--takes", nargs="+", default=["Take_006", "Take_007", "Take_012"])
    p.add_argument("--gallery-only", action="store_true")
    args = p.parse_args()
    if not args.gallery_only:
        for t in args.takes:
            render(t)
    TAKES = EXPERIMENT["takes"]
    gallery()
