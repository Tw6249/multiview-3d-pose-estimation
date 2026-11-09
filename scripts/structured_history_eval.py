"""Four-way geometry/history factorial ablation with shared scoring/rendering."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import ik_observation_gate_eval as shared
from g1_imitation.core.s45c_keypoint_ik import KeypointIKRetargeter
from g1_imitation.core.structured_history_ik import StructuredHistoryRetargeter


def factory(variant, config, model):
    if variant == "baseline":
        return KeypointIKRetargeter(model=model)
    return StructuredHistoryRetargeter(config, structured=variant in ("structured", "structured_history"),
                                       history=variant in ("history", "structured_history"), model=model)


def gallery():
    report = json.loads((shared.OUT / "report.json").read_text(encoding="utf-8"))
    fields = ("arms_reference_direction_deg", "arms_physics_direction_deg", "legs_physics_direction_deg",
              "support_foot_skating_m_s", "path_end_error_m", "ik_wall_p95_ms")
    rows = []
    for variant, metrics in report["summary"]["all"].items():
        rows.append("<tr><td>" + variant + "</td>" + "".join(f"<td>{metrics[key]:.4f}</td>" for key in fields) + "</tr>")
    videos = "".join(f'<section><h2>{take}</h2><video controls preload="metadata" src="{take}_comparison.mp4"></video>'
                     f'<img src="{take}_joint_traces.png" alt="Reference joint trajectories"></section>'
                     for take in report["results"] if (shared.OUT / f"{take}_comparison.mp4").exists())
    (shared.OUT / "index.html").write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>结构化几何与历史约束 IK</title><style>body{font:16px/1.7 system-ui;max-width:1600px;margin:28px auto;padding:20px;background:#f4f6f9;color:#243247}section{padding:20px;background:white;margin:24px 0}video,img{width:100%}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ddd}</style>
<h1>结构化几何与历史约束：四组相同条件对照</h1><p>2026-11-09 · 保存的多视角 RGB / Kalman 输入 · 同一 G1 与 SONIC v1.1，preview=0 · 仅仿真。</p>
<p>baseline 为原方法；structured 替换为肢段单位方向、弯曲关系和弱腕部相对位置；history 在原方法上增加历史先验与整帧关节速度/加速度 QP 约束；structured_history 同时启用。原骨盆/足部目标保留。没有按方法改模型、物理增益、时间或控制器输出。全部片段已被看过，属于探索而非新盲测。红色质量标记仅供诊断，本实验未启用上一轮观测门控。</p>
<section><h2>五段等权均值</h2><table><tr><th>方法</th><th>参考手臂 °</th><th>物理手臂 °</th><th>物理腿 °</th><th>脚滑 m/s</th><th>自身参考终点偏差 m</th><th>逐段 p95 耗时均值 ms</th></tr>'''
        + "".join(rows) + '</table><p><a href="report.json">完整逐段结果、配置和哈希</a></p></section><section><img src="summary.png"></section>'
        + videos + '</html>', encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--takes", nargs="+")
    parser.add_argument("--render", nargs="*", default=[])
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    shared.CONFIG_PATH = ROOT / "g1_imitation/config/structured_history_20261109.json"
    shared.OUT = ROOT / "outputs/structured_history_20261109"
    shared.COLORS = ("#3b6fb6", "#d88828", "#27845c", "#ad579e")
    config = json.loads(shared.CONFIG_PATH.read_text())
    if not args.render_only:
        shared.evaluate(args.takes or config["takes"], retargeter_factory=factory,
                        extra_source_files=(Path(__file__), ROOT / "g1_imitation/core/structured_history_ik.py",
                                            ROOT / "g1_imitation/core/retarget_experiments.py"))
    for take in args.render:
        shared.render(take)
    gallery()


if __name__ == "__main__":
    main()
