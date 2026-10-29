"""AdvancedTake single-person regression for Module S1b (multi-person association & target lock).

Question: does inserting S1b between S1 and S2 change the accepted single-person path? AdvancedTake has one subject
plus a bystander that YOLOX also detects (44310277 in every frame, 44310284 in 133-325 frames per take).
Input : data/advanced_take_raw/<Take>_Video_<cam>.mp4 (decoded in full; the frame count must equal the S1 cache),
        outputs/advanced_take/s1/<Take>_kp27.npz (accepted S1 cache: PTS and the single-box keypoints),
        outputs/advanced_take/cameras_rgbonly.json (RGB-only calibration; its world is not BVH-aligned, so only the
        Procrustes / per-frame PA / segment-direction numbers are meaningful), BVH via advanced_take_s2s3_eval.track_take.
Steps : (1) all YOLOX boxes + RTMW on each box -> outputs/advanced_take/s1b/<Take>_candidates.npz (cached); sanity
        check: the area x score pick among them reproduces the S1 cache keypoints;
        (2) per 30 Hz slot: S1b in auto mode (camera-ring capture volume, no designation) -> S2 -> S3 with the
        parameters of advanced_take_s2s3_eval; S2/S3 start when S1b locks the target;
        (3) the accepted path (track_take on the S1 cache, lag 6) on the same cameras; per-slot 3D difference and
        evaluate() metrics of both.
Output: outputs/advanced_take/s1b/regression.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from advanced_take_s2s3_eval import C_MIN, FPS, MAX_SPEED, REPROJ_PX, evaluate, load_cameras, track_take  # noqa: E402
from g1_imitation.core.s1_vision_tracker import RTMW27VisionTracker  # noqa: E402
from g1_imitation.core.s1b_person_association import (  # noqa: E402
    MultiPersonTargetTracker, ViewCandidates, camera_centers_m, extract_person_candidates)
from g1_imitation.core.s2_triangulation import MultiViewTriangulator  # noqa: E402
from g1_imitation.core.s3_temporal_filter import TemporalFilter  # noqa: E402

CAMS = ["44310274", "44310277", "44310281", "44310284"]
TAKES = ["Take_002", "Take_006", "Take_007", "Take_008", "Take_012"]
RAW = ROOT / "data" / "advanced_take_raw"
S1 = ROOT / "outputs" / "advanced_take" / "s1"
OUT = ROOT / "outputs" / "advanced_take" / "s1b"
CAMERAS = ROOT / "outputs" / "advanced_take" / "cameras_rgbonly.json"
LAG = 6
BVH_HZ = 120.0
KEYS = ("take_procrustes_mpjpe_mm", "per_frame_pa_mpjpe_mm", "segment_dir_err_mean_deg_after_rotation", "observed_fraction")


def extract(tracker, take: str) -> dict:
    path = OUT / f"{take}_candidates.npz"
    if not path.exists():
        cache = np.load(S1 / f"{take}_kp27.npz")
        pack = {}
        for cid in CAMS:
            cap = cv2.VideoCapture(str(RAW / f"{take}_Video_{cid}.mp4"))
            box, score, kp, ptr = [], [], [], [0]
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                vc = extract_person_candidates(tracker, [img])[0]
                box.append(vc.box_xyxy)
                score.append(vc.det_score)
                kp.append(vc.kp)
                ptr.append(ptr[-1] + len(vc.box_xyxy))
            cap.release()
            if len(ptr) - 1 != len(cache[f"{cid}_kp"]):
                raise RuntimeError(f"{take} {cid}: decoded {len(ptr) - 1} frames, S1 cache has {len(cache[f'{cid}_kp'])}")
            pack[f"{cid}_box"] = np.concatenate(box)
            pack[f"{cid}_score"] = np.concatenate(score)
            pack[f"{cid}_kp"] = np.concatenate(kp)
            pack[f"{cid}_ptr"] = np.array(ptr)
        np.savez_compressed(path, **pack)
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def sanity_vs_s1_cache(take: str, c: dict) -> dict:
    """Max |kp| difference between the area x score pick among all candidates and the accepted S1 cache."""
    cache = np.load(S1 / f"{take}_kp27.npz")
    worst, n = 0.0, 0
    for cid in CAMS:
        ptr = c[f"{cid}_ptr"]
        for i in range(len(ptr) - 1):
            a, b = ptr[i], ptr[i + 1]
            if b == a:
                continue
            bx = c[f"{cid}_box"][a:b]
            area = np.maximum(bx[:, 2] - bx[:, 0], 1.0) * np.maximum(bx[:, 3] - bx[:, 1], 1.0)
            j = a + int(np.argmax(area * c[f"{cid}_score"][a:b]))
            worst = max(worst, float(np.abs(c[f"{cid}_kp"][j] - cache[f"{cid}_kp"][i]).max()))
            n += 1
    return dict(frames_compared=n, max_abs_diff=worst)


def run_s1b(take: str, c: dict, P: np.ndarray, grid: np.ndarray, idx: dict) -> dict:
    T = len(grid)
    trk = MultiPersonTargetTracker()
    tri = flt = None
    raw, smooth = np.full((T, 27, 3), np.nan), np.full((T, 27, 3), np.nan)
    status, nviews, picked_xy = [], [], None
    empty = ViewCandidates(np.zeros((0, 4)), np.zeros(0), np.zeros((0, 27, 3)))
    for k in range(T):
        cands = []
        for cid in CAMS:
            i = idx[cid][k]
            if i < 0:
                cands.append(empty)
                continue
            a, b = c[f"{cid}_ptr"][i], c[f"{cid}_ptr"][i + 1]
            cands.append(ViewCandidates(c[f"{cid}_box"][a:b], c[f"{cid}_score"][a:b], c[f"{cid}_kp"][a:b]))
        res = trk.step(cands, P, grid[k] / BVH_HZ)
        if res.selected_now:
            tri = MultiViewTriangulator(c_min=C_MIN, reproj_outlier_threshold_px=REPROJ_PX,
                                        max_displacement_m=MAX_SPEED / FPS, use_consensus_search=True)
            flt = TemporalFilter(freq_hz=FPS, f_c_min=1.2, beta=0.04, f_c_d=1.0, z_nominal_m=0.0)
            picked_xy = res.track_centroids_m[res.target_id][:2].tolist()
        if tri is not None:
            r = tri.process_frame(res.keypoints_2d, P)
            raw[k] = r.points_3d_m
            smooth[k] = flt.process_frame(r.points_3d_m).points_3d_smooth_m
        status.append(res.status)
        nviews.append(len(res.target_views))
    return dict(raw=raw, smooth=smooth, status=status, nviews=nviews, picked_xy=picked_xy)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    P, _ = load_cameras(CAMERAS)
    cc = camera_centers_m(P)[:, :2]
    ring_c = cc.mean(axis=0)
    tracker = RTMW27VisionTracker(use_detector_every_frame=True)
    report = dict(cameras=str(CAMERAS.relative_to(ROOT)), lag_bvh_samples=LAG,
                  camera_ring_centre_xy_m=ring_c.tolist(),
                  camera_ring_mean_radius_m=float(np.linalg.norm(cc - ring_c, axis=1).mean()), takes={})
    for take in TAKES:
        c = extract(tracker, take)
        san = sanity_vs_s1_cache(take, c)
        tr0 = track_take(take, P, use_consensus=True, shifts=None, ref_lag=LAG)
        r = run_s1b(take, c, P, tr0["grid"], tr0["frame_idx"])
        tr1 = dict(tr0, raw=r["raw"], smooth=r["smooth"])
        body = slice(5, 17)
        both = np.isfinite(r["raw"][:, body]).all(axis=(1, 2)) & np.isfinite(tr0["raw"][:, body]).all(axis=(1, 2))
        diff = np.abs(r["raw"][:, body] - tr0["raw"][:, body]).max(axis=(1, 2))
        m = dict(sanity_pick_vs_s1_cache=san, slots=int(len(tr0["grid"])),
                 s1b_status={s: r["status"].count(s) for s in sorted(set(r["status"]))},
                 s1b_target_views_hist={str(n): int(r["nviews"].count(n)) for n in range(5)},
                 s1b_locked_at_xy_m=r["picked_xy"],
                 slots_both_observed=int(both.sum()),
                 slots_identical_body_3d=int((diff[both] < 1e-9).sum()),
                 body_3d_diff_m_max=float(diff[both].max()) if both.any() else None,
                 body_3d_diff_m_p99=float(np.percentile(diff[both], 99)) if both.any() else None,
                 accepted_raw={k: evaluate(tr0, "raw").get(k) for k in KEYS},
                 s1b_raw={k: evaluate(tr1, "raw").get(k) for k in KEYS},
                 accepted_filtered={k: evaluate(tr0, "smooth").get(k) for k in KEYS},
                 s1b_filtered={k: evaluate(tr1, "smooth").get(k) for k in KEYS})
        report["takes"][take] = m
        print(take, json.dumps({k: m[k] for k in ("sanity_pick_vs_s1_cache", "s1b_status", "s1b_target_views_hist",
                                                  "slots_both_observed", "slots_identical_body_3d", "body_3d_diff_m_max")}),
              flush=True)
        print("   raw  accepted", {k: round(v, 2) for k, v in m["accepted_raw"].items()}, "| s1b", {k: round(v, 2) for k, v in m["s1b_raw"].items()})
        print("   filt accepted", {k: round(v, 2) for k, v in m["accepted_filtered"].items()}, "| s1b", {k: round(v, 2) for k, v in m["s1b_filtered"].items()})
    (OUT / "regression.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
