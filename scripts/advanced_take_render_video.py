"""Render saved S1/S2 evidence against the BVH reference as a 1x-speed video. Never re-runs perception.

Inputs : outputs/advanced_take/s2s3/<tag>/<Take>.npz   (grid, raw 27-pt world tracks, BVH12; BVH already
         read `lag` samples later when the tag is a lag run), s1/<Take>_kp27.npz, cameras_selfcal.json, raw videos.
Output : outputs/advanced_take/video/<Take>_<tag>.mp4  (30 Hz slots, 1920x1080, H.264)
Units  : metres (world, BVH template-aligned Z-up), millimetres for errors, pixels in the 2048x1544 source frames.
Draw   : green = BVH12, orange = vision (UNFILTERED S2 output), yellow dots = S1 2D detections (conf >= 0.35).
         Unobserved joints are not drawn and not imputed. The error curve is never clamped.
Honesty: world error mixes joint-definition offsets, calibration scale and clock residual; BVH is a solved skeleton.
         The clock offset is the one estimated from the calibration takes, not a hardware synchronisation.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from g1_imitation.core import adv_eval as ae  # noqa: E402

CAMS = ["44310274", "44310277", "44310281", "44310284"]
OUT = ROOT / "outputs" / "advanced_take"
RAW = ROOT / "data" / "advanced_take_raw"
FFMPEG = "ffmpeg"
W, H = 1920, 1080
HEAD, FOOT = 78, 40
TILE_W, TILE_H = 600, 452
EDGES12 = [(0, 1), (6, 7), (0, 6), (1, 7), (0, 2), (2, 4), (1, 3), (3, 5), (6, 8), (8, 10), (7, 9), (9, 11)]
BVH_BGR, VIS_BGR, DET_BGR = (113, 204, 46), (0, 140, 255), (0, 235, 255)
BG, INK, MUTED = (247, 244, 241), (72, 52, 30), (120, 104, 82)   # BGR
FONT = {n: ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", n) for n in (16, 18, 22, 30)}
C_MIN = 0.35


def put_text(img: np.ndarray, items: list[tuple]) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for xy, s, size, bgr in items:
        d.text(xy, s, font=FONT[size], fill=(bgr[2], bgr[1], bgr[0]))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


class Reader:
    """Forward-only decoder: frame(i) returns the frame with that index (frames are consumed in order)."""

    def __init__(self, path: Path):
        self.cap = cv2.VideoCapture(str(path))
        self.i = -1
        self.cur = None

    def frame(self, i: int):
        if i < self.i:
            raise ValueError("forward-only reader")
        while self.i < i:
            ok, f = self.cap.read()
            if not ok:
                return None
            self.i, self.cur = self.i + 1, f
        return self.cur


def project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    """X (N,3) world -> (N,2) pixels; NaN where non-finite or behind the camera."""
    Xh = np.hstack([X, np.ones((len(X), 1))])
    p = Xh @ P.T
    out = np.full((len(X), 2), np.nan)
    ok = np.isfinite(p).all(1) & (p[:, 2] > 1e-6)
    out[ok] = p[ok, :2] / p[ok, 2:3]
    return out


def draw_skel(img, uv, color, scale, th=2, r=3):
    pts = [None if not np.isfinite(q).all() else (int(round(q[0] * scale)), int(round(q[1] * scale))) for q in uv]
    for a, b in EDGES12:
        if pts[a] is not None and pts[b] is not None:
            cv2.line(img, pts[a], pts[b], color, th, cv2.LINE_AA)
    for p in pts:
        if p is not None:
            cv2.circle(img, p, r, color, -1, cv2.LINE_AA)


class View3D:
    """Oblique orthographic view that follows the BVH pelvis; the floor grid is schematic (registration unverified)."""

    def __init__(self, w, h, floor_z, yaw_deg=38.0, pitch_deg=24.0, px_per_m=215.0):
        self.w, self.h, self.floor_z, self.s = w, h, floor_z, px_per_m
        y, p = np.radians(yaw_deg), np.radians(pitch_deg)
        self.Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
        self.cp, self.sp = np.cos(p), np.sin(p)
        self.c = np.zeros(3)

    def pix(self, X):
        v = (np.asarray(X) - self.c) @ self.Rz.T
        up = v[..., 2] * self.cp + v[..., 1] * self.sp
        return np.stack([self.w * 0.5 + v[..., 0] * self.s, self.h * 0.72 - up * self.s], -1)

    def render(self, vis, bvh, err_mm):
        img = np.full((self.h, self.w, 3), (250, 247, 244), np.uint8)
        hips = bvh[[6, 7]]
        if np.isfinite(hips).all():
            self.c = np.array([hips[:, 0].mean(), hips[:, 1].mean(), self.floor_z])
        g0 = np.floor((self.c[:2] - 3.0) / 0.5) * 0.5
        for k in range(14):
            for a, b in ((np.array([g0[0] + 0.5 * k, g0[1], self.floor_z]), np.array([g0[0] + 0.5 * k, g0[1] + 6.5, self.floor_z])),
                         (np.array([g0[0], g0[1] + 0.5 * k, self.floor_z]), np.array([g0[0] + 6.5, g0[1] + 0.5 * k, self.floor_z]))):
                pa, pb = self.pix(a), self.pix(b)
                cv2.line(img, tuple(int(v) for v in pa), tuple(int(v) for v in pb), (226, 220, 212), 1, cv2.LINE_AA)
        for j in range(12):                                   # per-joint error segments
            if np.isfinite(vis[j]).all() and np.isfinite(bvh[j]).all():
                cv2.line(img, tuple(int(v) for v in self.pix(vis[j])), tuple(int(v) for v in self.pix(bvh[j])),
                         (60, 60, 220), 2, cv2.LINE_AA)
        for pts, col in ((bvh, BVH_BGR), (vis, VIS_BGR)):
            uv = self.pix(pts)
            draw_skel(img, uv, col, 1.0, th=3, r=4)
        return img


def err_panel(w, h, e_mm, k, fps=30.0):
    img = np.full((h, w, 3), (250, 247, 244), np.uint8)
    L, R_, T, B = 58, 16, 34, 34
    ymax = max(20.0, float(np.ceil(np.nanmax(e_mm) / 20.0) * 20.0))
    n = len(e_mm)

    def X(i): return int(L + (w - L - R_) * i / max(1, n - 1))
    def Y(v): return int(h - B - (h - T - B) * v / ymax)
    for v in np.arange(0, ymax + 1, 20 if ymax <= 120 else 40):
        cv2.line(img, (L, Y(v)), (w - R_, Y(v)), (232, 228, 222), 1)
    pts = [(X(i), Y(v)) for i, v in enumerate(e_mm) if np.isfinite(v)]
    if len(pts) > 1:
        cv2.polylines(img, [np.array(pts, np.int32)], False, VIS_BGR, 2, cv2.LINE_AA)
    mean = float(np.nanmean(e_mm))
    cv2.line(img, (L, Y(mean)), (w - R_, Y(mean)), (150, 140, 128), 1, cv2.LINE_AA)
    cv2.line(img, (X(k), T), (X(k), h - B), INK, 1, cv2.LINE_AA)
    if np.isfinite(e_mm[k]):
        cv2.circle(img, (X(k), Y(e_mm[k])), 5, INK, -1, cv2.LINE_AA)
    items = [((8, Y(v) - 10), f"{int(v)}", 16, MUTED) for v in np.arange(0, ymax + 1, 20 if ymax <= 120 else 40)]
    items += [((L, h - B + 6), "0 s", 16, MUTED), ((w - R_ - 44, h - B + 6), f"{(n - 1) / fps:.0f} s", 16, MUTED),
              ((w - R_ - 215, 8), f"全程均值 {mean:.1f} mm（细线）", 16, MUTED)]
    return put_text(img, items)


def render_take(take: str, tag: str, lag_note: str, P: np.ndarray, floor_pct: float = 1.0):
    z = np.load(OUT / "s2s3" / tag / f"{take}.npz")
    raw, bvh12, grid = z["raw"][:, 5:17], z["bvh12"], z["grid"]
    d = np.load(OUT / "s1" / f"{take}_kp27.npz")
    from g1_imitation.core.bvh import parse_bvh_and_fk
    bvh = parse_bvh_and_fk(RAW / "bvh" / f"{take}_Skeleton0.bvh")
    _, fidx = ae.build_slot_grid({c: d[f"{c}_pts_s"] for c in CAMS}, bvh["n_motion_frames"])
    T = len(grid)
    e = np.linalg.norm(raw - bvh12, axis=-1)                          # (T,12) m
    e_mm = np.full(T, np.nan)
    has = np.isfinite(e).any(1)
    e_mm[has] = np.nanmean(e[has], axis=1) * 1000
    ank = bvh12[:, 10:12, 2]
    floor_z = float(np.nanpercentile(ank, floor_pct)) - 0.08
    view = View3D(720, 540, floor_z)
    readers = {c: Reader(RAW / f"{take}_Video_{c}.mp4") for c in CAMS}
    odir = OUT / "video"
    odir.mkdir(parents=True, exist_ok=True)
    dest = odir / f"{take}_{tag}.mp4"
    ff = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                           "-r", "30", "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                           "-movflags", "+faststart", str(dest)], stdin=subprocess.PIPE)
    sc = TILE_W / 2048.0
    jn = ae.JOINTS12
    for k in range(T):
        canvas = np.full((H, W, 3), BG, np.uint8)
        labels = []
        for ci, c in enumerate(CAMS):
            x0, y0 = (ci % 2) * TILE_W, HEAD + (ci // 2) * TILE_H
            i = int(fidx[c][k])
            fr = readers[c].frame(i) if i >= 0 else None
            if fr is None:
                tile = np.full((TILE_H, TILE_W, 3), 40, np.uint8)
                labels.append(((x0 + 10, y0 + 8), f"{c}  本槽位无帧", 16, (255, 255, 255)))
            else:
                tile = cv2.resize(fr, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
                kp = d[f"{c}_kp"][i]
                for j in range(12):
                    if kp[5 + j, 2] >= C_MIN:
                        cv2.circle(tile, (int(kp[5 + j, 0] * sc), int(kp[5 + j, 1] * sc)), 4, DET_BGR, 1, cv2.LINE_AA)
                draw_skel(tile, project(P[ci], bvh12[k]), BVH_BGR, sc, th=2, r=3)
                draw_skel(tile, project(P[ci], raw[k]), VIS_BGR, sc, th=2, r=3)
                labels.append(((x0 + 10, y0 + 8), f"机位 {c}", 16, (255, 255, 255)))
            cv2.rectangle(tile, (0, 0), (190, 30), (0, 0, 0), -1)
            canvas[y0:y0 + TILE_H, x0:x0 + TILE_W] = tile
        rx = 2 * TILE_W + 0
        canvas[HEAD:HEAD + 540, rx:rx + 720] = view.render(raw[k], bvh12[k], e_mm[k])
        canvas[HEAD + 540:HEAD + 540 + 324, rx:rx + 720] = err_panel(720, 324, e_mm, k)
        ej = np.where(np.isfinite(e[k]), e[k], -1.0)
        worst = f"最大偏差 {jn[int(ej.argmax())]} {ej.max() * 1000:.0f} mm" if ej.max() >= 0 else "本帧无可比关节"
        nobs = int(np.isfinite(e[k]).sum())
        items = [((16, 10), f"{take} · 留出集 · 三角化(未滤波)  vs  BVH", 30, INK),
                 ((16, 48), "绿=BVH   橙=视觉3D   黄圈=S1二维检测   红线=逐关节偏差", 18, MUTED),
                 ((rx + 14, HEAD + 8), f"t = {k / 30.0:5.2f} s   本帧 MPJPE {e_mm[k]:.0f} mm" if np.isfinite(e_mm[k]) else f"t = {k / 30.0:5.2f} s   本帧无可比关节", 22, INK),
                 ((rx + 14, HEAD + 40), f"{worst}   可比关节 {nobs}/12", 16, MUTED),
                 ((rx + 14, HEAD + 540 + 2), "逐帧 MPJPE（未钳位，mm）", 16, MUTED),
                 ((16, H - FOOT + 8), f"自标定相机，BVH 仅作参照；{lag_note}。误差含关节定义偏移与标定/时钟残差，不等于纯感知误差。仿真评估，无实机。", 16, MUTED)]
        canvas = put_text(canvas, labels + items)
        ff.stdin.write(canvas.tobytes())
    ff.stdin.close()
    ff.wait()
    for r in readers.values():
        r.cap.release()
    return dest, dict(slots=int(T), mean_mpjpe_mm=float(np.nanmean(e_mm)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--takes", nargs="+", default=["Take_006", "Take_007", "Take_008"])
    ap.add_argument("--tag", default="selfcal_lag6")
    ap.add_argument("--cameras", default=str(OUT / "cameras_selfcal.json"))
    args = ap.parse_args()
    cams = json.loads(Path(args.cameras).read_text(encoding="utf-8"))["cameras"]
    P = np.array([np.array(cams[c]["P"]) for c in CAMS])
    m = json.loads((OUT / "s2s3" / args.tag / "metrics.json").read_text(encoding="utf-8"))
    lag = int(m.get("ref_lag_bvh_samples", 0))
    note = f"BVH 已按估计偏移 +{lag * 1000 / 120:.0f} ms 对齐（由标定用 Take 估计，非硬件同步）" if lag else "BVH 使用旧的末帧对齐时间映射（未补偿）"
    for take in args.takes:
        dest, info = render_take(take, args.tag, note, P)
        print(take, dest, info, flush=True)


if __name__ == "__main__":
    main()
