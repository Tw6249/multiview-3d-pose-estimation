"""MAMMA multi-person session: S1 (all candidates) -> S1b target lock -> S2 -> S3 (-> S4+S5 keypoint IK -> S6 SONIC).

Input : data/public_mamma_session (4 cameras, 120 frames at 30 Hz, two people pushing, one falls; bystanders).
        The target is designated by an operator click (camera, u, v) on the first frame (`--target`, repeatable:
        one run per click). There are NO 3D labels: every number here is a consistency check, not accuracy.
Output: outputs/mamma_multiperson/<run>/ (metrics.json, arrays.npz, overlay_*.jpg) and outputs/mamma_multiperson/summary.json.
Baseline: the current single-person S1 state machine (YOLOX + reprojected-track crop per camera) -> S2 -> S3, chained
        as in scripts/run_full_simulation_demo.py, with the same S2/S3 parameters.
Checks: status per frame; frame-to-frame jump of the S2 torso centre (a person switch shows as a jump); bone-length
        CV; S2 reprojection error; torso colour (HSV) of the crops used (independent of the geometry); separation
        between the runs' targets (two runs locked on different people must never coincide); closed loop (sim).
Protocol: S2/S3/S4+S5 start (and restart) when S1b reports `selected_now`; before that the slot is NaN. Invalid frames
        give S2 zero-score keypoints, so S3 holds its last estimate (S1b failure behaviour).
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from g1_imitation.core.s1_vision_tracker import RTMW27VisionTracker  # noqa: E402
from g1_imitation.core.s1b_person_association import MultiPersonTargetTracker, extract_person_candidates  # noqa: E402
from g1_imitation.core.s2_triangulation import MultiViewTriangulator  # noqa: E402
from g1_imitation.core.s3_temporal_filter import TemporalFilter  # noqa: E402

SES = ROOT / "data" / "public_mamma_session"
OUT = ROOT / "outputs" / "mamma_multiperson"
FPS = 30.0
C_MIN = 0.35            # pipeline_config.json s2_triangulation.min_2d_confidence
REPROJ_PX = 15.0        # pipeline_config.json s2_triangulation.reprojection_error_threshold_px
MAX_SPEED = 25.0        # m/s, as in advanced_take_s2s3_eval.py
BODY = list(range(5, 17))
BONES = {"L_upperarm": (5, 7), "R_upperarm": (6, 8), "L_forearm": (7, 9), "R_forearm": (8, 10), "L_thigh": (11, 13),
         "R_thigh": (12, 14), "L_shank": (13, 15), "R_shank": (14, 16), "shoulder_w": (5, 6), "hip_w": (11, 12)}
EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def load_session():
    cams = json.loads((SES / "cameras.json").read_text(encoding="utf-8"))["cameras"]
    names = sorted(cams)
    P = np.stack([np.array(cams[c]["K"]) @ np.hstack([np.array(cams[c]["R_world_to_camera"]),
                                                      np.array(cams[c]["t_world_to_camera_m"])[:, None]]) for c in names])
    frames = [json.loads(s) for s in (SES / "frames.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    return names, P, frames


def read_images(fr, names):
    # imdecode: cv2.imread cannot open the non-ASCII workspace path on Windows
    return [cv2.imdecode(np.fromfile(str(SES / fr["views"][c]["image"]), np.uint8), cv2.IMREAD_COLOR) for c in names]


def new_s2_s3():
    return (MultiViewTriangulator(c_min=C_MIN, reproj_outlier_threshold_px=REPROJ_PX, max_displacement_m=MAX_SPEED / FPS,
                                  use_consensus_search=True),
            TemporalFilter(freq_hz=FPS, f_c_min=1.2, beta=0.04, f_c_d=1.0, z_nominal_m=0.0))


def torso_hsv(img, kp):
    q = kp[[5, 6, 12, 11]]
    if (q[:, 2] < C_MIN).any():
        return None
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(m, q[:, :2].astype(np.int32), 1)
    if m.sum() < 50:
        return None
    return np.median(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[m > 0], axis=0)


def summarize(raw, reproj, status, views, hsv):
    T = len(raw)
    ok = np.isfinite(raw[:, BODY]).all(axis=(1, 2))
    torso = np.nanmean(raw[:, [5, 6, 11, 12]], axis=1)
    jumps = [float(np.linalg.norm(torso[k] - torso[k - 1])) for k in range(1, T) if ok[k] and ok[k - 1]]
    bl = {n: np.linalg.norm(raw[ok, a] - raw[ok, b], axis=1) for n, (a, b) in BONES.items()}
    v_series = np.array([np.nan if h is None else h[2] for h in hsv])
    return dict(
        frames=T, status_counts={s: status.count(s) for s in sorted(set(status))},
        frames_with_3d_body=int(ok.sum()),
        views_hist={str(n): int(sum(1 for v in views if len(v) == n)) for n in range(5)},
        torso_jump_m_max=float(max(jumps)) if jumps else None, torso_jump_m_p95=float(np.percentile(jumps, 95)) if jumps else None,
        torso_jump_frames_gt_0p15m=int(sum(j > 0.15 for j in jumps)),
        bone_cv_pct={n: float(np.std(x) / np.mean(x) * 100) for n, x in bl.items()} if ok.sum() > 2 else None,
        body_reproj_px_median=float(np.nanmedian(reproj[:, BODY])) if np.isfinite(reproj).any() else None,
        body_reproj_px_p95=float(np.nanpercentile(reproj[:, BODY], 95)) if np.isfinite(reproj).any() else None,
        torso_V_median=float(np.nanmedian(v_series)) if np.isfinite(v_series).any() else None,
        torso_V_p5_p95=[float(np.nanpercentile(v_series, 5)), float(np.nanpercentile(v_series, 95))] if np.isfinite(v_series).any() else None,
        torso_V_series=[None if not np.isfinite(x) else round(float(x), 1) for x in v_series],
        torso_centre_series=[None if not np.isfinite(c).all() else np.round(c, 3).tolist() for c in torso])


def run_s1b(names, P, frames, tracker, click):
    trk = MultiPersonTargetTracker()
    trk.designate_pixel(*click)
    tri = flt = None
    T = len(frames)
    raw, smooth = np.full((T, 27, 3), np.nan), np.full((T, 27, 3), np.nan)
    reproj = np.full((T, 27), np.nan)
    status, views, hsv, boxes, hyps, margin = [], [], [], [], [], []
    ms = dict(detect=[], pose=[], associate=[], n_candidates=[])
    for k, fr in enumerate(frames):
        imgs = read_images(fr, names)
        t0 = time.perf_counter()
        det = [tracker.detect_person_candidates(im) for im in imgs]
        t1 = time.perf_counter()
        cands = extract_person_candidates(_Cached(tracker, det), imgs)
        t2 = time.perf_counter()
        res = trk.step(cands, P, fr["timestamp_ns"] * 1e-9)
        ms["detect"].append((t1 - t0) * 1e3)
        ms["pose"].append((t2 - t1) * 1e3)
        ms["associate"].append(res.associate_ms)
        ms["n_candidates"].append(sum(len(c.box_xyxy) for c in cands))
        if res.selected_now:
            tri, flt = new_s2_s3()
        if tri is not None:
            r2 = tri.process_frame(res.keypoints_2d, P)
            raw[k], reproj[k] = r2.points_3d_m, r2.reprojection_errors_px
            smooth[k] = flt.process_frame(r2.points_3d_m, dt=1.0 / FPS).points_3d_smooth_m
        status.append(res.status)
        views.append(res.target_views)
        boxes.append(res.crop_bboxes_xyxy)
        margin.append(res.identity_margin_m)
        hs = [torso_hsv(imgs[v], res.keypoints_2d[v]) for v in res.target_views]
        hs = [h for h in hs if h is not None]
        hsv.append(np.median(hs, axis=0) if hs else None)
        hyps.append(dict(n=len(res.hypotheses), views=[h.n_views for h in res.hypotheses],
                         tracks=list(res.hypothesis_track_ids),
                         centres={int(i): np.round(c, 3).tolist() for i, c in res.track_centroids_m.items()},
                         cand_boxes=[c.box_xyxy.tolist() for c in cands]))
    return dict(raw=raw, smooth=smooth, reproj=reproj, status=status, views=views, hsv=hsv, boxes=np.array(boxes),
                hyps=hyps, margin=margin, ms=ms)


class _Cached:
    """Feeds already computed YOLOX boxes to `extract_person_candidates` (so detection and pose time separately)."""

    def __init__(self, tracker, det):
        self.tracker, self.det, self.i = tracker, det, 0
        self.DETECT_MODE = tracker.DETECT_MODE

    def detect_person_candidates(self, img):
        d = self.det[self.i]
        self.i += 1
        return d

    def infer_batch_from_bboxes(self, *a, **k):
        return self.tracker.infer_batch_from_bboxes(*a, **k)


def run_baseline(names, P, frames):
    """Current single-person S1 (TRACK/DETECT state machine with 3D feedback) -> S2 -> S3."""
    tracker = RTMW27VisionTracker(use_detector_every_frame=True)
    tri, flt = new_s2_s3()
    T = len(frames)
    raw, smooth = np.full((T, 27, 3), np.nan), np.full((T, 27, 3), np.nan)
    reproj = np.full((T, 27), np.nan)
    status, views, hsv, boxes, s1_ms = [], [], [], [], []
    for k, fr in enumerate(frames):
        imgs = read_images(fr, names)
        t0 = time.perf_counter()
        r1 = tracker.process_frame(imgs, P, dt=1.0 / FPS)
        s1_ms.append((time.perf_counter() - t0) * 1e3)
        r2 = tri.process_frame(r1.keypoints_2d, P)
        r3 = flt.process_frame(r2.points_3d_m, dt=1.0 / FPS)
        tracker.update_3d_state(r3.points_3d_smooth_m, dt=1.0 / FPS)
        raw[k], smooth[k], reproj[k] = r2.points_3d_m, r3.points_3d_smooth_m, r2.reprojection_errors_px
        status.append(r1.mode_used)
        views.append(tuple(range(len(names))))
        boxes.append(r1.crop_bboxes_xyxy)
        hs = [h for h in (torso_hsv(imgs[v], r1.keypoints_2d[v]) for v in range(len(names))) if h is not None]
        hsv.append(np.median(hs, axis=0) if hs else None)
    return dict(raw=raw, smooth=smooth, reproj=reproj, status=status, views=views, hsv=hsv, boxes=np.array(boxes),
                ms=dict(s1_process_frame=s1_ms))


def timing_ab(names, P, frames, tracker, click, reps):
    """Interleaved wall time per frame (images already decoded): A = current S1 `process_frame` (YOLOX x 4 + RTMW
    on 4 crops, TRACK state machine fed back untimed); B = YOLOX x 4 + RTMW on every candidate + S1b. The order of
    A and B alternates per frame so both see the same machine load. Returns per-frame arrays over all repetitions."""
    a_ms, b_ms = [], []
    for rep in range(reps):
        base = RTMW27VisionTracker(use_detector_every_frame=True)
        tri, flt = new_s2_s3()
        trk = MultiPersonTargetTracker()
        trk.designate_pixel(*click)
        for k, fr in enumerate(frames):
            imgs = read_images(fr, names)
            for which in ((0, 1) if (k + rep) % 2 == 0 else (1, 0)):
                t0 = time.perf_counter()
                if which == 0:
                    r1 = base.process_frame(imgs, P, dt=1.0 / FPS)
                    a_ms.append((time.perf_counter() - t0) * 1e3)
                    r3 = flt.process_frame(tri.process_frame(r1.keypoints_2d, P).points_3d_m, dt=1.0 / FPS)
                    base.update_3d_state(r3.points_3d_smooth_m, dt=1.0 / FPS)
                else:
                    trk.step(extract_person_candidates(tracker, imgs), P, fr["timestamp_ns"] * 1e-9)
                    b_ms.append((time.perf_counter() - t0) * 1e3)
    a, b = np.array(a_ms), np.array(b_ms)
    q = lambda x: dict(p50=float(np.median(x)), p95=float(np.percentile(x, 95)), max=float(x.max()))
    return dict(reps=reps, frames=len(a), s1_current=q(a), s1_all_candidates_plus_s1b=q(b), paired_difference=q(b - a))


def closed_loop(smooth):
    """S4+S5 keypoint IK (causal, 30 Hz slots) -> 50 Hz hold -> S6 SONIC physics (simulation only)."""
    from g1_imitation.core.constants import load_config
    from g1_imitation.core.s45c_keypoint_ik import KeypointIKRetargeter, causal_upsample, retarget_sequence
    from g1_imitation.core.s6_mujoco_closed_loop import MuJoCoClosedLoopVerifier, SonicPolicy
    from advanced_take_gmr_control import anchor_heading
    first = int(np.argmax(np.isfinite(smooth).all(axis=(1, 2))))
    pts = smooth[first:]
    t_slot = np.arange(len(pts)) / FPS
    rt = KeypointIKRetargeter()
    out = retarget_sequence(pts, t_slot, rt)
    _, q, dq, _ = causal_upsample(t_slot, out["q"], out["dq"], out["observed"], "hold", rt.lo, rt.hi, dt=0.02)
    q, dq, _ = anchor_heading(q, dq)
    cfg = load_config()["s6_closed_loop"]
    res = dict(first_slot=first, ref_pelvis_z_min_m=float(q[:, 2].min()),
               ref_joint_speed_max_rad_s=float(np.abs(np.diff(q[:, 7:], axis=0) / 0.02).max()), runs={})
    for ctrl in ("sonic_v11", "sonic_low_latency"):
        r = MuJoCoClosedLoopVerifier(policy=SonicPolicy(cfg[ctrl])).run_stage_6b_physics(
            q, dq, ik_dt=0.02, control_dt=0.02, settle_steps=25, preview_s=0.0)
        low = np.flatnonzero(r.qpos_hist[:, 2] < cfg["survival_min_pelvis_z_m"])
        res["runs"][ctrl] = dict(survived=bool(r.survived), joint_rmse_rad=float(r.joint_rmse_rad),
                                 max_tilt_deg=float(r.max_tilt_deg),
                                 first_pelvis_below_min_s=None if len(low) == 0 else float(low[0] * 0.02))
    return res


def overlays(names, P, frames, run, odir, every=4, per_sheet=8):
    sel = list(range(0, len(frames), every))
    for s in range(0, len(sel), per_sheet):
        rows = []
        for k in sel[s:s + per_sheet]:
            imgs = read_images(frames[k], names)
            tiles = []
            for v, im in enumerate(imgs):
                im = im.copy()
                for b in run.get("hyps", [{}] * len(frames))[k].get("cand_boxes", [[]] * len(names))[v]:
                    cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (160, 160, 160), 2)
                b = run["boxes"][k, v]
                if np.isfinite(b).all() and (v in run["views"][k]):
                    cv2.rectangle(im, (int(b[0]) - 4, int(b[1]) - 4), (int(b[2]) + 4, int(b[3]) + 4), (0, 200, 255), 5)
                X = run["raw"][k]
                if np.isfinite(X[BODY]).any():
                    Xh = np.c_[np.nan_to_num(X), np.ones(27)]
                    uvw = Xh @ P[v].T
                    uv = uvw[:, :2] / uvw[:, 2:3]
                    for a, c in EDGES:
                        if np.isfinite(X[[a, c]]).all():
                            cv2.line(im, tuple(int(x) for x in uv[a]), tuple(int(x) for x in uv[c]), (0, 0, 255), 5)
                im = cv2.resize(im, (480, 270))
                cv2.putText(im, f"{names[v]} f{k} {run['status'][k][:4]}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                tiles.append(im)
            rows.append(np.hstack(tiles))
        cv2.imencode(".jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tofile(str(odir / f"overlay_{s // per_sheet}.jpg"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", nargs=4, action="append", metavar=("NAME", "CAM", "U", "V"),
                    default=None, help="run name and operator click on the first frame, e.g. P1 A001 440 440")
    ap.add_argument("--no-closed-loop", action="store_true")
    ap.add_argument("--no-overlays", action="store_true")
    ap.add_argument("--timing-reps", type=int, default=0,
                    help="only run the interleaved A/B wall-time comparison this many times over the session")
    a = ap.parse_args()
    targets = a.target or [["P1_beige", "A001", "440", "440"], ["P0_cap", "A001", "740", "400"]]
    names, P, frames = load_session()
    OUT.mkdir(parents=True, exist_ok=True)
    tracker = RTMW27VisionTracker(use_detector_every_frame=True)
    if a.timing_reps > 0:
        _, cam, u, v = targets[0]
        t = timing_ab(names, P, frames, tracker, (names.index(cam), float(u), float(v)), a.timing_reps)
        (OUT / "timing_ab.json").write_text(json.dumps(t, indent=1), encoding="utf-8")
        print(json.dumps(t, indent=1))
        return
    runs, summary = {}, {}
    for name, cam, u, v in targets:
        r = run_s1b(names, P, frames, tracker, (names.index(cam), float(u), float(v)))
        runs[name] = r
        m = summarize(r["raw"], r["reproj"], r["status"], r["views"], r["hsv"])
        m["click"] = [cam, float(u), float(v)]
        m["identity_margin_m_min_valid"] = float(np.nanmin([x for x, s in zip(r["margin"], r["status"]) if s == "TRACKING"]))
        m["latency_ms"] = {k: dict(p50=float(np.median(x)), p95=float(np.percentile(x, 95)), max=float(np.max(x)))
                           for k, x in r["ms"].items()}
        m["status_series"] = "".join(s[0] for s in r["status"])
        summary[name] = m
    base = run_baseline(names, P, frames)
    runs["baseline_s1_track"] = base
    summary["baseline_s1_track"] = summarize(base["raw"], base["reproj"], base["status"], base["views"], base["hsv"])
    summary["baseline_s1_track"]["latency_ms"] = {k: dict(p50=float(np.median(x)), p95=float(np.percentile(x, 95)),
                                                          max=float(np.max(x))) for k, x in base["ms"].items()}
    keys = [t[0] for t in targets]
    if len(keys) >= 2:
        ca = np.nanmean(runs[keys[0]]["raw"][:, [5, 6, 11, 12]], axis=1)
        cb = np.nanmean(runs[keys[1]]["raw"][:, [5, 6, 11, 12]], axis=1)
        d = np.linalg.norm(ca - cb, axis=1)
        summary["separation_between_first_two_targets_m"] = dict(min=float(np.nanmin(d)), p5=float(np.nanpercentile(d, 5)),
                                                                 frames=int(np.isfinite(d).sum()))
    if not a.no_closed_loop:
        for name in keys + ["baseline_s1_track"]:
            summary[name]["closed_loop"] = closed_loop(runs[name]["smooth"])
    for name, r in runs.items():
        odir = OUT / name
        odir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(odir / "arrays.npz", raw=r["raw"], smooth=r["smooth"], reproj=r["reproj"], boxes=r["boxes"])
        (odir / "metrics.json").write_text(json.dumps(summary[name], indent=1), encoding="utf-8")
        if "hyps" in r:
            (odir / "hypotheses.json").write_text(json.dumps(r["hyps"]), encoding="utf-8")
        if not a.no_overlays:
            overlays(names, P, frames, r, odir)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for name, m in summary.items():
        if not isinstance(m, dict) or "status_counts" not in m:
            print(name, m)
            continue
        print(f"== {name}: {m['status_counts']} views {m['views_hist']} jump max {m['torso_jump_m_max']:.3f} m "
              f"(>0.15 m: {m['torso_jump_frames_gt_0p15m']}) reproj {m['body_reproj_px_median']:.1f}/{m['body_reproj_px_p95']:.1f} px "
              f"torso V {m['torso_V_median']:.0f} [{m['torso_V_p5_p95'][0]:.0f},{m['torso_V_p5_p95'][1]:.0f}]")
        print("   bone CV %:", {k: round(x, 1) for k, x in (m["bone_cv_pct"] or {}).items()})
        if "status_series" in m:
            print("   status:", m["status_series"], "| min margin", round(m["identity_margin_m_min_valid"], 3))
        if "latency_ms" in m:
            print("   latency:", {k: (round(x["p50"], 1), round(x["p95"], 1)) for k, x in m["latency_ms"].items()})
        if "closed_loop" in m:
            print("   closed loop:", m["closed_loop"])


if __name__ == "__main__":
    main()
