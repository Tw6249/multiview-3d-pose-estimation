"""Showcase videos: raw 4-camera views with our 2D skeleton | 3D skeleton | G1 reference | G1 physics tracking.

Output: outputs/showcase_videos/<case>.mp4 (1920x1080 H.264 at the source frame rate) and <case>.json (run summary).
Cases (`--cases`, default all):
  mamma_p1, mamma_p0   data/public_mamma_session (outdoor, two people, one falls). S1 on every YOLOX box + S1b, the
                       target clicked on the first frame (P1 standing / P0 falls).
  advtake_Take_006/007/008  AdvancedTake raw videos, accepted single-person path WITHOUT S1b: the S1 cache (YOLOX + the
                       largest box per camera) -> S2 -> S3 on the 30 Hz slot grid (cameras_rgbonly.json, lag 6).
  mpi_s1_seq1          MPI-INF-3DHP S1/Seq1, cameras 0/2/5/7, 18-21 s (25 fps, frames resized to the 1024 px
                       calibration of data/public_datasets/mpi_benchmark). S1 on every box + S1b (auto selection).
  blender_clutter      outputs/showcase_videos/blender_clutter/session (scripts/blender_clutter_scene.py): synthetic
                       person among clutter. S1 on every box + S1b (auto selection). Truth = SMPL-X joints.
Downstream (every case): S3 One-Euro at the frame rate -> S4+S5 causal keypoint IK per frame -> 50 Hz hold ->
  S6 SONIC (`s6_closed_loop.default_policy`) in MuJoCo. Simulation only; the robot starts at the origin facing +X
  (reference re-anchored to its first heading), panels 2/3 are viewed from the same body-relative direction as panel 1.
Draw: 2D = the keypoints S2 used (score >= 0.35); grey boxes = all person candidates, yellow = the crop used.
  Unobserved points are not drawn. Status text is the S1b status of that frame.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from advanced_take_render_rgb_control import (  # noqa: E402
    BG, FOOT, H, HEAD, INK, MUTED, PH, PW, SKEL_VIEW_AZ, TH, TW, W, Skel3D, draw27, g1_renderer, ref_view_offset_deg)
from advanced_take_render_video import FFMPEG, Reader, put_text  # noqa: E402
from g1_imitation.core.constants import load_config  # noqa: E402
from g1_imitation.core.s1_vision_tracker import RTMW27VisionTracker  # noqa: E402
from g1_imitation.core.s1b_person_association import MultiPersonTargetTracker, extract_person_candidates  # noqa: E402
from g1_imitation.core.s2_triangulation import MultiViewTriangulator  # noqa: E402
from g1_imitation.core.s3_temporal_filter import TemporalFilter  # noqa: E402
from g1_imitation.core.s6_mujoco_closed_loop import G1PhysicsPlant  # noqa: E402

OUT = ROOT / "outputs" / "showcase_videos"
C_MIN = 0.35
ADV_CAMS = ["44310274", "44310277", "44310281", "44310284"]
SMPLX_FOR_COCO = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}


# ----------------------------------------------------------------------------------------------- frame sources
class SessionSource:
    """Folder with cameras.json + frames.jsonl + images (MAMMA session layout); synchronized frames."""

    def __init__(self, folder: Path, title: str):
        self.folder, self.title = folder, title
        cams = json.loads((folder / "cameras.json").read_text(encoding="utf-8"))["cameras"]
        self.names = sorted(cams)
        self.P = np.stack([np.array(cams[c]["K"]) @ np.hstack([np.array(cams[c]["R_world_to_camera"]),
                                                               np.array(cams[c]["t_world_to_camera_m"])[:, None]])
                           for c in self.names])
        self.size = [(int(cams[c]["width"]), int(cams[c]["height"])) for c in self.names]
        self.frames = [json.loads(s) for s in (folder / "frames.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
        ts = np.array([f["timestamp_ns"] for f in self.frames], np.float64) * 1e-9
        self.t = ts - ts[0]
        self.fps = float(round(1.0 / np.median(np.diff(ts))))

    def images(self):
        for fr in self.frames:
            yield [cv2.imdecode(np.fromfile(str(self.folder / fr["views"][c]["image"]), np.uint8), cv2.IMREAD_COLOR)
                   for c in self.names]


class MpiSource:
    """MPI-INF-3DHP S1/Seq1 videos (2048 px, 25 fps) resized to the 1024 px calibration; frames from start_s to end_s."""
    CAMS = ["0", "2", "5", "7"]

    def __init__(self, start_s: float, end_s: float):
        base = ROOT / "data" / "public_datasets" / "mpi_benchmark" / "inputs" / "s1_exercise"
        cams = json.loads((base / "cameras.json").read_text(encoding="utf-8"))
        src = json.loads((base / "source.json").read_text(encoding="utf-8"))
        if src["sequence"] != "S1/Seq1":
            raise RuntimeError("calibration file is not for S1/Seq1")
        self.title = f"MPI-INF-3DHP S1/Seq1 · {start_s:.0f}–{end_s:.0f} s · 机位 {'/'.join(self.CAMS)}"
        self.names = self.CAMS
        self.fps = float(src["fps"])
        self.start = int(round(start_s * self.fps))
        n = int(round((end_s - start_s) * self.fps))
        cc = cams["cameras"]
        self.P = np.stack([np.array(cc[c]["K"]) @ np.hstack([np.array(cc[c]["R_world_to_camera"]),
                                                             np.array(cc[c]["t_world_to_camera_m"])[:, None]])
                           for c in self.names])
        self.size = [(int(cc[c]["width"]), int(cc[c]["height"])) for c in self.names]
        self.videos = [ROOT / "data" / "public_datasets" / "mpi_inf_3dhp" / "S1" / "Seq1" / "imageSequence" / f"video_{c}.avi"
                       for c in self.names]
        self.t = np.arange(n) / self.fps
        self.world_axes = cams["world_axes"]

    def images(self):
        caps = [cv2.VideoCapture(str(p)) for p in self.videos]
        for cap in caps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start)
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != self.start:
                raise RuntimeError("video seek failed")
        for _ in range(len(self.t)):
            imgs = []
            for cap, (w, h) in zip(caps, self.size):
                ok, f = cap.read()
                if not ok:
                    raise RuntimeError("video ended early")
                imgs.append(cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA))
            yield imgs
        for cap in caps:
            cap.release()


class AdvSource:
    """AdvancedTake raw videos on the accepted 30 Hz slot grid (per-camera PTS end alignment, lag 6)."""

    def __init__(self, take: str):
        from advanced_take_s2s3_eval import load_cameras, track_take
        self.take, self.names = take, ADV_CAMS
        self.title = f"AdvancedTake {take}（青瞳 Chingmu 实验室采集，单人）"
        self.P, _ = load_cameras(ROOT / "outputs" / "advanced_take" / "cameras_rgbonly.json")
        self.tr = track_take(take, self.P, use_consensus=True, shifts=None, ref_lag=6)
        self.fidx = self.tr["frame_idx"]
        self.t = (self.tr["grid"] - self.tr["grid"][0]) / 120.0
        self.fps = 30.0
        self.size = [(2048, 1544)] * 4
        self.cache = np.load(ROOT / "outputs" / "advanced_take" / "s1" / f"{take}_kp27.npz")

    def images(self):
        rd = [Reader(ROOT / "data" / "advanced_take_raw" / f"{self.take}_Video_{c}.mp4") for c in self.names]
        for k in range(len(self.t)):
            yield [r.frame(int(self.fidx[c][k])) if self.fidx[c][k] >= 0 else None for r, c in zip(rd, self.names)]


# ----------------------------------------------------------------------------------------------- perception
def s2_s3(fps: float):
    return (MultiViewTriangulator(c_min=C_MIN, reproj_outlier_threshold_px=15.0, max_displacement_m=25.0 / fps,
                                  use_consensus_search=True),
            TemporalFilter(freq_hz=fps, f_c_min=1.2, beta=0.04, f_c_d=1.0, z_nominal_m=0.0))


def perceive_s1b(src, tracker, click=None) -> dict:
    T, V = len(src.t), len(src.names)
    trk = MultiPersonTargetTracker()
    if click is not None:
        trk.designate_pixel(src.names.index(click[0]), click[1], click[2])
    out = dict(kp2d=np.zeros((T, V, 27, 3)), box=np.full((T, V, 4), np.nan), cands=[], status=[],
               raw=np.full((T, 27, 3), np.nan), smooth=np.full((T, 27, 3), np.nan), s1_ms=[])
    tri = flt = None
    for k, imgs in enumerate(src.images()):
        t0 = time.perf_counter()
        cands = extract_person_candidates(tracker, imgs)
        res = trk.step(cands, src.P, src.t[k])
        out["s1_ms"].append((time.perf_counter() - t0) * 1e3)
        if res.selected_now:
            tri, flt = s2_s3(src.fps)
        if tri is not None:
            r2 = tri.process_frame(res.keypoints_2d, src.P)
            out["raw"][k] = r2.points_3d_m
            out["smooth"][k] = flt.process_frame(r2.points_3d_m, dt=1.0 / src.fps).points_3d_smooth_m
        out["kp2d"][k], out["box"][k] = res.keypoints_2d, res.crop_bboxes_xyxy
        out["cands"].append([c.box_xyxy for c in cands])
        out["status"].append(res.status)
    out["mode"] = "S1 全部候选 + S1b 多人关联与目标锁定" + ("（操作者点选目标）" if click else "（自动选择）")
    return out


def ground_from_feet(per: dict, fps: float) -> float:
    """One-off registration for a world whose floor is not at z = 0: lift the S2 track so the 1st percentile of the
    heel/toe points over the clip is z = 0, then re-run S3 on it. Uses our own detections only (no labels), but the
    whole clip, so it is not causal. Returns the lift in metres."""
    dz = -float(np.nanpercentile(per["raw"][:, [17, 18, 19, 20, 21, 22], 2], 1))
    per["raw"] = per["raw"] + np.array([0.0, 0.0, dz])
    flt = TemporalFilter(freq_hz=fps, f_c_min=1.2, beta=0.04, f_c_d=1.0, z_nominal_m=0.0)
    first = int(np.argmax(np.isfinite(per["raw"]).any(axis=(1, 2))))
    per["smooth"] = np.full_like(per["raw"], np.nan)
    for k in range(first, len(per["raw"])):
        per["smooth"][k] = flt.process_frame(per["raw"][k], dt=1.0 / fps).points_3d_smooth_m
    return dz


def perceive_accepted(src: AdvSource) -> dict:
    T, V = len(src.t), len(src.names)
    out = dict(kp2d=np.zeros((T, V, 27, 3)), box=np.full((T, V, 4), np.nan), cands=[None] * T, status=["S1"] * T,
               raw=src.tr["raw"], smooth=src.tr["smooth"], s1_ms=[])
    for v, c in enumerate(src.names):
        for k in range(T):
            i = int(src.fidx[c][k])
            if i >= 0 and src.cache[f"{c}_det_count"][i] > 0:
                out["kp2d"][k, v] = src.cache[f"{c}_kp"][i]
                out["box"][k, v] = src.cache[f"{c}_bbox"][i]
    out["mode"] = "S1 单人（每机位取最大框，无多人处理）"
    return out


# ----------------------------------------------------------------------------------------------- robot
def robot(t: np.ndarray, smooth: np.ndarray) -> dict:
    from advanced_take_gmr_control import anchor_heading
    from g1_imitation.core.s45c_keypoint_ik import KeypointIKRetargeter, causal_upsample, retarget_sequence
    from g1_imitation.core.s6_mujoco_closed_loop import MuJoCoClosedLoopVerifier, SonicPolicy
    cfg = load_config()["s6_closed_loop"]
    policy = cfg["default_policy"]
    first = int(np.argmax(np.isfinite(smooth).all(axis=(1, 2))))
    rt = KeypointIKRetargeter()
    out = retarget_sequence(smooth[first:], t[first:], rt)
    t50, q, dq, _ = causal_upsample(t[first:], out["q"], out["dq"], out["observed"], "hold", rt.lo, rt.hi, dt=0.02)
    q, dq, _ = anchor_heading(q, dq)
    r = MuJoCoClosedLoopVerifier(policy=SonicPolicy(cfg[policy])).run_stage_6b_physics(
        q, dq, ik_dt=0.02, control_dt=0.02, settle_steps=25, preview_s=0.0)
    low = np.flatnonzero(r.qpos_hist[:, 2] < cfg["survival_min_pelvis_z_m"])
    return dict(t50=t50, q_ref=q, q_phys=r.qpos_hist, policy=policy, survived=bool(r.survived),
                joint_rmse_deg=float(np.degrees(r.joint_rmse_rad)), max_tilt_deg=float(r.max_tilt_deg),
                fall_s=None if len(low) == 0 else float(t50[min(low[0], len(t50) - 1)]),
                ik_solve_ms_p50=float(np.median(out["solve_ms"][out["observed"]])))


# ----------------------------------------------------------------------------------------------- drawing
def fit_tile(img, size):
    """Letterbox `img` into a TW x TH tile; returns the tile and (scale, x0, y0) from source pixels to tile pixels."""
    w, h = size
    s = min(TW / w, TH / h)
    tw, th = int(round(w * s)), int(round(h * s))
    x0, y0 = (TW - tw) // 2, (TH - th) // 2
    tile = np.full((TH, TW, 3), 30, np.uint8)
    if img is not None:
        tile[y0:y0 + th, x0:x0 + tw] = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    return tile, (s, x0, y0)


def to_tile(uv, tf):
    s, x0, y0 = tf
    return uv * s + np.array([x0, y0])


def render_case(name: str, src, per: dict, rob: dict, extra: list[str]):
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{name}.mp4"
    ff = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                           "-r", f"{src.fps:g}", "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                           "-movflags", "+faststart", str(dest)], stdin=subprocess.PIPE)
    model = G1PhysicsPlant().kin_model
    d_ref, d_phys = mujoco.MjData(model), mujoco.MjData(model)
    rend, cam = g1_renderer(model)
    view = Skel3D()
    track = per["smooth"]
    t50, q_ref, q_phys = rob["t50"], rob["q_ref"], rob["q_phys"]
    view_off = ref_view_offset_deg(track, src.t, t50, q_ref)
    pel_xy = 0.5 * (track[:, 11, :2] + track[:, 12, :2])
    surv = "全程未倒地" if rob["survived"] else f"{rob['fall_s']:.1f} s 倒地"
    for k, imgs in enumerate(src.images()):
        canvas = np.full((H, W, 3), BG, np.uint8)
        labels = []
        for v, img in enumerate(imgs):
            tile, tf = fit_tile(img, src.size[v])
            for b in (per["cands"][k][v] if per["cands"][k] is not None else []):
                p = to_tile(np.asarray(b, float).reshape(2, 2), tf).astype(int)
                cv2.rectangle(tile, tuple(p[0]), tuple(p[1]), (170, 170, 170), 1, cv2.LINE_AA)
            if np.isfinite(per["box"][k, v]).all():
                p = to_tile(per["box"][k, v].reshape(2, 2), tf).astype(int)
                cv2.rectangle(tile, tuple(p[0]), tuple(p[1]), (0, 210, 255), 2, cv2.LINE_AA)
            kp = per["kp2d"][k, v]
            uv = np.where((kp[:, 2] >= C_MIN)[:, None], kp[:, :2], np.nan)
            draw27(tile, to_tile(uv, tf), th=2)
            cv2.rectangle(tile, (0, 0), (170, 24), (0, 0, 0), -1)
            canvas[HEAD:HEAD + TH, v * TW:(v + 1) * TW] = tile
            labels.append(((v * TW + 8, HEAD + 2), f"机位 {src.names[v]}" if img is not None else f"{src.names[v]} 无帧",
                           16, (255, 255, 255)))
        y0 = HEAD + TH
        canvas[y0:y0 + PH, 0:PW] = view.render(track[k], pel_xy[:k + 1])
        j = int(np.clip(np.searchsorted(t50, src.t[k]), 0, len(t50) - 1))
        started = src.t[k] >= t50[0] - 0.02
        for dd, q, x0 in ((d_ref, q_ref, PW), (d_phys, q_phys, 2 * PW)):
            dd.qpos[:] = q[min(j, len(q) - 1)]
            mujoco.mj_forward(model, dd)
            cam.lookat[:] = [dd.qpos[0], dd.qpos[1], 0.65]
            cam.azimuth = SKEL_VIEW_AZ + view_off
            rend.update_scene(dd, camera=cam)
            canvas[y0:y0 + PH, x0:x0 + PW] = cv2.cvtColor(rend.render(), cv2.COLOR_RGB2BGR)
        nobs = int(np.isfinite(track[k]).all(1).sum())
        st = per["status"][k]
        pz = float(q_phys[min(j, len(q_phys) - 1), 2])
        items = [((16, 6), f"{src.title}", 30, INK),
                 ((16, 46), f"{per['mode']} → S2 三角化 → S3 滤波 → 关键点 IK 重定向 → {rob['policy']} 物理跟踪", 18, MUTED),
                 ((12, y0 + 8), f"① 视觉 3D 骨架（S3 后，观测 {nobs}/27）  t={src.t[k]:5.2f} s  状态 {st}", 18, INK),
                 ((12, y0 + 34), "橙=左  蓝=右  灰线=骨盆轨迹；上方：黄框=所用裁剪" +
                  ("，灰框=全部人体候选" if per["cands"][k] is not None else "（单人路径只保存所选框）"), 16, MUTED),
                 ((PW + 12, y0 + 8), "② G1 重定向参考（运动学）", 18, (255, 255, 255)),
                 ((PW + 12, y0 + 34), "与 ① 同一相对人体的视角" + ("" if started else "（标定/锁定中，显示首帧）"),
                  16, (225, 225, 225)),
                 ((2 * PW + 12, y0 + 8), f"③ 机器人实际跟踪（MuJoCo 物理）· {surv}", 18, (255, 255, 255)),
                 ((2 * PW + 12, y0 + 34), f"骨盆高 {pz:.2f} m · 关节 RMSE {rob['joint_rmse_deg']:.1f}°", 16, (225, 225, 225)),
                 ((16, H - FOOT + 8), "仿真，非实机。" + "  ".join(extra), 16, MUTED)]
        canvas = put_text(canvas, labels + items)
        ff.stdin.write(canvas.tobytes())
    ff.stdin.close()
    ff.wait()
    rend.close()
    return dest


def truth_error(per: dict, mesh_npz: Path) -> dict:
    """Synthetic case only: S2 raw body joints vs the SMPL-X joints that drove the mesh (joint definitions differ:
    COCO surface landmarks vs SMPL-X joint centres, so this mixes a definition offset into the error)."""
    j = np.load(mesh_npz)["joints"]
    raw = per["raw"][:, list(SMPLX_FOR_COCO)]
    gt = j[: len(raw)][:, list(SMPLX_FOR_COCO.values())]
    e = np.linalg.norm(raw - gt, axis=-1)
    ok = np.isfinite(e)
    return dict(mpjpe_mm=float(e[ok].mean() * 1000), median_mm=float(np.median(e[ok]) * 1000),
                p95_mm=float(np.percentile(e[ok], 95) * 1000), observed_fraction=float(ok.mean()),
                note="12 body joints, S2 output (unfiltered) vs SMPL-X joint centres; includes landmark-definition offsets")


def build(name: str, tracker):
    extra = []
    truth = None
    if name in ("mamma_p1", "mamma_p0"):
        src = SessionSource(ROOT / "data" / "public_mamma_session", "MAMMA 户外多人（两人推搡，P0 被推倒）")
        click = ("A001", 440.0, 440.0) if name == "mamma_p1" else ("A001", 740.0, 400.0)
        src.title += " · 目标 " + ("P1（站立者）" if name == "mamma_p1" else "P0（倒地者）")
        per = perceive_s1b(src, tracker, click)
        extra.append(f"目标由操作者在 {click[0]} 首帧点选 ({click[1]:.0f}, {click[2]:.0f})。无 3D 标注。")
        if name == "mamma_p0":
            extra.append("倒地动作超出模仿范围，尚无动作域门限，机器人随之倒地。")
    elif name.startswith("advtake_"):
        src = AdvSource(name[len("advtake_"):])
        per = perceive_accepted(src)
        extra.append("RGB 自标定（不用 BVH）；相机同步为末帧对齐假设 + 滞后 6。")
    elif name == "mpi_s1_seq1":
        src = MpiSource(18.0, 21.0)
        per = perceive_s1b(src, tracker)
        dz = ground_from_feet(per, src.fps)
        per["ground_lift_m"] = dz
        extra.append(f"官方标定；该世界系地面不在 z=0，按本段脚点 1% 分位整体上移 {dz * 100:.1f} cm（一次性、非因果配准）。")
    elif name == "blender_clutter":
        folder = OUT / "blender_clutter" / "session"
        src = SessionSource(folder, "Blender 合成：四机位 + 杂物（纸箱、桌椅、锥桶、三脚架、盆栽、货架、衣架）")
        per = perceive_s1b(src, tracker)
        truth = truth_error(per, OUT / "blender_clutter" / "mesh_Take_002.npz")
        extra.append(f"合成数据，相机标定与同步为真值；S2 对 SMPL-X 关节 MPJPE {truth['mpjpe_mm']:.0f} mm（含关节定义差）。")
    else:
        raise ValueError(name)
    rob = robot(src.t, per["smooth"])
    dest = render_case(name, src, per, rob, extra)
    ok = np.isfinite(per["raw"][:, 5:17]).all(axis=(1, 2))
    feet = per["raw"][:, [15, 16, 17, 18, 19, 20, 21, 22], 2]
    summary = dict(video=str(dest.relative_to(ROOT)), frames=len(src.t), fps=src.fps, perception=per["mode"],
                   status_counts={s: per["status"].count(s) for s in sorted(set(per["status"]))},
                   frames_with_12_body_joints=int(ok.sum()),
                   foot_z_m_p1_p5=[float(np.nanpercentile(feet, 1)), float(np.nanpercentile(feet, 5))],
                   s1_s1b_ms_p50=float(np.median(per["s1_ms"])) if per["s1_ms"] else None,
                   ground_lift_m=per.get("ground_lift_m"),
                   robot={k: v for k, v in rob.items() if k not in ("t50", "q_ref", "q_phys")}, truth=truth)
    (OUT / f"{name}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    return summary


def write_index() -> Path:
    """INDEX.md in the output folder from every <case>.json present."""
    rows = ["| 视频 | 帧数 / fps | 感知 | 状态 | 机器人（" + load_config()["s6_closed_loop"]["default_policy"] + "） | 备注 |",
            "|---|---|---|---|---|---|"]
    for p in sorted(OUT.glob("*.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        r = s["robot"]
        rob = ("未倒地" if r["survived"] else f"{r['fall_s']:.1f} s 倒地") + f"，关节 RMSE {r['joint_rmse_deg']:.1f}°"
        note = []
        if s.get("ground_lift_m"):
            note.append(f"地面配准 +{s['ground_lift_m'] * 100:.1f} cm")
        if s.get("truth"):
            note.append(f"对 SMPL-X 关节 MPJPE {s['truth']['mpjpe_mm']:.0f} mm（中位 {s['truth']['median_mm']:.0f}）")
        st = ", ".join(f"{k} {v}" for k, v in s["status_counts"].items())
        rows.append(f"| `{Path(s['video']).name}` | {s['frames']} / {s['fps']:g} | {s['perception']} | {st} | {rob} | {'；'.join(note)} |")
    dest = OUT / "INDEX.md"
    dest.write_text("# 渲染视频索引（scripts/render_showcase_videos.py 生成）\n\n每个视频：上排四路原始画面 + 我们的 2D 骨架；"
                    "下排 ① 视觉 3D 骨架（S3 后）② G1 重定向参考 ③ MuJoCo 物理跟踪。仿真，非实机。\n\n"
                    + "\n".join(rows) + "\n", encoding="utf-8")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["mamma_p1", "mamma_p0", "advtake_Take_006", "advtake_Take_007",
                                                   "advtake_Take_008", "mpi_s1_seq1", "blender_clutter"])
    ap.add_argument("--index-only", action="store_true")
    a = ap.parse_args()
    if a.index_only:
        print(write_index())
        return
    tracker = RTMW27VisionTracker(use_detector_every_frame=True)
    for name in a.cases:
        t0 = time.perf_counter()
        s = build(name, tracker)
        print(name, f"{time.perf_counter() - t0:.0f}s", json.dumps({k: s[k] for k in ("status_counts", "foot_z_m_p1_p5")},
              ensure_ascii=False), json.dumps(s["robot"]), json.dumps(s["truth"]) if s["truth"] else "", flush=True)
    print(write_index())


if __name__ == "__main__":
    main()
