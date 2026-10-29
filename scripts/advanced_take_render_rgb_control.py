"""Render the pure-RGB chain (4 RGB views -> 27-pt -> S4+S5 G1 reference -> SONIC MuJoCo physics) as a video.

Nothing is re-run here: inputs are saved outputs only, and no BVH data is drawn.
Inputs : s2s3/<tag>/<Take>.npz (27-pt RGB-world tracks), s1/<Take>_kp27.npz, cameras_rgbonly.json, raw videos,
         `--retargeter gmr` : control/gmr_smplx_<tag>/<Take>.npz + metrics.json (offline SMPL-X + GMR, SONIC v1.1);
         `--retargeter kpik`: control/kpik_ll0_smooth_<tag>/<Take>.npz + keypoint_ik/report_ll0.json (default causal
                              keypoint IK on the S3-filtered track, SONIC low_latency, no added delay), made by
                              s6_keypoint_ik_eval.py --inputs smooth --modes hold --save-ctrl sonic_low_latency@0.00 --tag ll0.
Output : outputs/advanced_take/video/<Take>_rgb_<gmr|kpik>_control.mp4 (30 Hz video slots, 1920x1080, H.264).
Time   : each video slot shows the nearest 50 Hz control sample (<= 10 ms apart). The video slot clock is the
         end-alignment hypothesis; the control timeline is the same slot clock resampled, not a hardware clock.
Draw   : unobserved points are not drawn; no smoothing is added for display.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from advanced_take_render_video import CAMS, FFMPEG, OUT, RAW, Reader, project, put_text  # noqa: E402
from g1_imitation.core import adv_eval as ae  # noqa: E402
from g1_imitation.core.s6_mujoco_closed_loop import G1PhysicsPlant  # noqa: E402

W, H, HEAD, FOOT = 1920, 1080, 78, 40
TW, TH = 480, 362
PW, PH = 640, H - HEAD - TH - FOOT
BVH_DT = 1.0 / 120.0
EDGES27 = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (11, 12), (5, 11), (6, 12), (5, 7), (7, 9), (6, 8), (8, 10),
           (11, 13), (13, 15), (12, 14), (14, 16), (15, 19), (15, 17), (15, 18), (17, 18), (16, 22), (16, 20),
           (16, 21), (20, 21), (9, 23), (9, 24), (23, 24), (10, 25), (10, 26), (25, 26)]
LEFT = {1, 3, 5, 7, 9, 11, 13, 15, 17, 18, 19, 23, 24}
L_BGR, R_BGR, C_BGR = (0, 140, 255), (235, 120, 40), (90, 90, 90)
BG, INK, MUTED = (247, 244, 241), (72, 52, 30), (120, 104, 82)


def edge_color(a, b):
    if a in LEFT and b in LEFT:
        return L_BGR
    if a not in LEFT and b not in LEFT and a > 4 and b > 4:
        return R_BGR
    return C_BGR


def draw27(img, uv, scale=1.0, th=2):
    ok = np.isfinite(uv).all(1)
    p = np.round(np.where(ok[:, None], uv, 0) * scale).astype(int)
    for a, b in EDGES27:
        if ok[a] and ok[b]:
            cv2.line(img, tuple(p[a]), tuple(p[b]), edge_color(a, b), th, cv2.LINE_AA)
    for j in np.flatnonzero(ok):
        cv2.circle(img, tuple(p[j]), th + 1, (255, 255, 255) if j < 5 else edge_color(j, j), -1, cv2.LINE_AA)


SKEL_YAW, SKEL_PITCH = 38.0, 22.0
SKEL_VIEW_AZ = 90.0 - SKEL_YAW   # world direction (deg) the panel-1 view looks along, MuJoCo azimuth convention


def hip_yaw_deg(left, right):
    lat = left - right
    return np.degrees(np.arctan2(lat[..., 1], lat[..., 0])) - 90.0


def ref_view_offset_deg(track, t_vis, t50, q_ref):
    """Constant yaw (deg) between the reference world and the 27-pt world (the control runs re-anchor the reference
    to heading 0). Median over frames of reference pelvis yaw minus human hip-line yaw; one value per take, so a
    heading error of the physics run stays visible."""
    from scipy.spatial.transform import Rotation
    ok = np.isfinite(track[:, [11, 12]]).all((1, 2)) & (t_vis >= t50[0]) & (t_vis <= t50[-1])
    j = np.clip(np.searchsorted(t50, t_vis[ok]), 0, len(q_ref) - 1)
    yr = Rotation.from_quat(q_ref[j][:, [4, 5, 6, 3]]).as_euler("zyx", degrees=True)[:, 0]
    d = (yr - hip_yaw_deg(track[ok, 11], track[ok, 12]) + 180.0) % 360.0 - 180.0
    return float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(d))))))


class Skel3D:
    """Oblique orthographic view following the RGB pelvis; ground z = 0 is the RGB-estimated plane."""

    def __init__(self, yaw=SKEL_YAW, pitch=SKEL_PITCH, s=230.0):
        y, p = np.radians(yaw), np.radians(pitch)
        self.Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
        self.cp, self.sp, self.s, self.c = np.cos(p), np.sin(p), s, np.zeros(3)

    def pix(self, X):
        v = (np.asarray(X) - self.c) @ self.Rz.T
        return np.stack([PW * 0.5 + v[..., 0] * self.s, PH * 0.80 - (v[..., 2] * self.cp + v[..., 1] * self.sp) * self.s], -1)

    def render(self, X, trail):
        img = np.full((PH, PW, 3), (250, 247, 244), np.uint8)
        pel = 0.5 * (X[11] + X[12])
        if np.isfinite(pel).all():
            self.c = np.array([pel[0], pel[1], 0.0])
        g0 = np.floor((self.c[:2] - 3.0) / 0.5) * 0.5
        for k in range(14):
            for a, b in (([g0[0] + 0.5 * k, g0[1], 0], [g0[0] + 0.5 * k, g0[1] + 6.5, 0]),
                         ([g0[0], g0[1] + 0.5 * k, 0], [g0[0] + 6.5, g0[1] + 0.5 * k, 0])):
                pa, pb = self.pix(np.array(a, float)), self.pix(np.array(b, float))
                cv2.line(img, tuple(pa.astype(int)), tuple(pb.astype(int)), (226, 220, 212), 1, cv2.LINE_AA)
        tr = trail[np.isfinite(trail).all(1)]
        if len(tr) > 1:
            cv2.polylines(img, [self.pix(np.c_[tr, np.zeros(len(tr))]).astype(np.int32)], False, (180, 170, 160), 2, cv2.LINE_AA)
        draw27(img, self.pix(X), th=3)
        return img


def g1_renderer(model):
    # display-only framebuffer size; this model instance is used for drawing, never stepped
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, PW)
    model.vis.global_.offheight = max(model.vis.global_.offheight, PH)
    r = mujoco.Renderer(model, height=PH, width=PW)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance, cam.elevation, cam.azimuth = 2.6, -SKEL_PITCH, 135.0
    return r, cam


RETARGETERS = {  # name -> (control folder, npz pattern, panel-2 title, video suffix, drawn 3D track, controller label)
    "gmr": ("gmr_smplx_{tag}", "{take}.npz", "② G1 运动学参考（EasyMocap SMPL-X 拟合 + GMR，离线）", "rgb_gmr_control",
            "raw", "SONIC v1.1"),
    "kpik": ("kpik_ll0_smooth_{tag}", "{take}.npz", "② G1 运动学参考（关键点 IK，因果逐帧）", "rgb_kpik_control",
             "smooth", "SONIC low_latency"),
}


def run_metrics(retargeter: str, folder: str, take: str) -> tuple[bool, float]:
    """(survived, joint RMSE deg vs own reference) of the saved physics run."""
    if retargeter == "gmr":
        e = json.loads((OUT / "control" / folder / "metrics.json").read_text(encoding="utf-8"))["takes"][take]["stage6b"]
        return bool(e["survived"]), float(e["joint_rmse_deg_vs_own_ref"])
    r = json.loads((OUT / "keypoint_ik" / "report_ll0.json").read_text(encoding="utf-8"))[f"smooth|{take}|hold"]
    e = r["runs"]["sonic_low_latency@0.00"]
    return bool(e["survived"]), float(np.degrees(e["joint_rmse_rad"]))


def render(take: str, tag: str, retargeter: str = "gmr"):
    folder, pattern, ref_title, suffix, track, ctrl_label = RETARGETERS[retargeter]
    folder = folder.format(tag=tag)
    z = np.load(OUT / "s2s3" / tag / f"{take}.npz")
    raw, grid = z[track], z["grid"]
    d = np.load(OUT / "s1" / f"{take}_kp27.npz")
    ctl = np.load(OUT / "control" / folder / pattern.format(take=take))
    t50, q_ref, q_phys = ctl["t50"], ctl["q_ref"], ctl["qpos_phys"]
    survived, rmse_deg = run_metrics(retargeter, folder, take)
    lag = int(json.loads((OUT / "s2s3" / tag / "metrics.json").read_text(encoding="utf-8"))["ref_lag_bvh_samples"])
    cams = json.loads((OUT / "cameras_rgbonly.json").read_text(encoding="utf-8"))["cameras"]
    P = np.array([np.array(cams[c]["P"]) for c in CAMS])
    n_motion = int(grid[-1] + 1)  # the slot grid labels end at the last motion sample; only the labels are reused
    _, fidx = ae.build_slot_grid({c: d[f"{c}_pts_s"] for c in CAMS}, n_motion)
    if not np.array_equal(_, grid):
        raise RuntimeError("slot grid mismatch")
    t_vis = (grid + lag) * BVH_DT
    plant = G1PhysicsPlant()
    model = plant.kin_model
    d_ref, d_phys = mujoco.MjData(model), mujoco.MjData(model)
    rend, cam = g1_renderer(model)
    readers = {c: Reader(RAW / f"{take}_Video_{c}.mp4") for c in CAMS}
    view = Skel3D()
    odir = OUT / "video"
    odir.mkdir(parents=True, exist_ok=True)
    dest = odir / f"{take}_{suffix}.mp4"
    ff = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                           "-r", "30", "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                           "-movflags", "+faststart", str(dest)], stdin=subprocess.PIPE)
    sc = TW / 2048.0
    view_off = ref_view_offset_deg(raw, t_vis, t50, q_ref)
    pel_xy = 0.5 * (raw[:, 11, :2] + raw[:, 12, :2])
    surv_txt = "存活" if survived else "倒地"
    for k in range(len(grid)):
        canvas = np.full((H, W, 3), BG, np.uint8)
        labels = []
        for ci, c in enumerate(CAMS):
            x0 = ci * TW
            i = int(fidx[c][k])
            fr = readers[c].frame(i) if i >= 0 else None
            if fr is None:
                tile = np.full((TH, TW, 3), 40, np.uint8)
            else:
                tile = cv2.resize(fr, (TW, TH), interpolation=cv2.INTER_AREA)
                draw27(tile, project(P[ci], raw[k]), scale=sc, th=2)
            cv2.rectangle(tile, (0, 0), (150, 26), (0, 0, 0), -1)
            canvas[HEAD:HEAD + TH, x0:x0 + TW] = tile
            labels.append(((x0 + 8, HEAD + 3), f"机位 {c}" if fr is not None else f"{c} 无帧", 16, (255, 255, 255)))
        y0 = HEAD + TH
        canvas[y0:y0 + PH, 0:PW] = view.render(raw[k], pel_xy[:k + 1])
        j = int(np.clip(np.searchsorted(t50, t_vis[k]), 0, len(t50) - 1))
        in_ctl = t50[0] - 0.02 <= t_vis[k] <= t50[-1] + 0.02
        for dd, q, x0 in ((d_ref, q_ref, PW), (d_phys, q_phys, 2 * PW)):
            dd.qpos[:] = q[min(j, len(q) - 1)]
            mujoco.mj_forward(model, dd)
            cam.lookat[:] = [dd.qpos[0], dd.qpos[1], 0.65]
            cam.azimuth = SKEL_VIEW_AZ + view_off   # same body-relative view as panel 1
            rend.update_scene(dd, camera=cam)
            canvas[y0:y0 + PH, x0:x0 + PW] = cv2.cvtColor(rend.render(), cv2.COLOR_RGB2BGR)
        nobs = int(np.isfinite(raw[k]).all(1).sum())
        pz = float(q_phys[min(j, len(q_phys) - 1), 2])
        items = [((16, 8), f"{take} · 纯 RGB 链路：4 路相机 → 三角化 27 点 → 重定向 → SONIC 物理仿真", 30, INK),
                 ((16, 48), "无标记、无 BVH 输入；世界系重力/地面来自脚点拟合，尺度来自骨盆高度先验 0.92 m", 18, MUTED),
                 ((12, y0 + 8), f"① 视觉 3D 骨架（27 点，观测 {nobs}/27）  t={k / 30:5.2f} s", 18, INK),
                 ((12, y0 + 34), "橙=左  蓝=右  灰线=骨盆轨迹", 16, MUTED),
                 ((PW + 12, y0 + 8), ref_title, 18, (255, 255, 255)),
                 ((PW + 12, y0 + 34), "视角与 ① 相同（相对人体朝向）", 16, (225, 225, 225)),
                 ((2 * PW + 12, y0 + 8), f"③ {ctrl_label} 闭环物理（MuJoCo）· 全程{surv_txt}", 18, (255, 255, 255)),
                 ((2 * PW + 12, y0 + 34), f"骨盆高 {pz:.2f} m" + ("" if in_ctl else "  （控制窗口外，显示端点）"), 16, (225, 225, 225)),
                 ((16, H - FOOT + 8), f"仿真，非实机。SONIC 对自身参考的关节 RMSE {rmse_deg:.1f}°；相机同步为末帧对齐假设；"
                                      + ("视觉骨架为 S3 滤波后（即重定向的输入）。" if track == "smooth" else "视觉骨架未滤波。"), 16, MUTED)]
        canvas = put_text(canvas, labels + items)
        ff.stdin.write(canvas.tobytes())
    ff.stdin.close()
    ff.wait()
    rend.close()
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--takes", nargs="+", default=["Take_006", "Take_007", "Take_008"])
    ap.add_argument("--tag", default="rgbonly_lag6")
    ap.add_argument("--retargeter", choices=sorted(RETARGETERS), default="gmr")
    args = ap.parse_args()
    for t in args.takes:
        print(t, render(t, args.tag, args.retargeter), flush=True)


if __name__ == "__main__":
    main()
