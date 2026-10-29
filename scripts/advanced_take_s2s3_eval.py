"""S2 (triangulation) + S3 (One-Euro) on fresh AdvancedTake S1 2D, compared with the BVH skeleton.

Inputs : S1 npz (advanced_take_s1_extract.py), cameras json (self-cal by default, or the legacy BVH-fitted
         cameras for the calibration-source comparison), BVH files (reference only, never an input to S2/S3).
Outputs: outputs/advanced_take/s2s3/<tag>/<Take>.npz  (raw/filtered 27-pt world tracks, BVH12, covariances)
         outputs/advanced_take/s2s3/<tag>/metrics.json
World  : BVH template-aligned Z-up metres (room registration unverified). Time: legacy end-alignment, UNVERIFIED.
Metrics: world MPJPE, take-level Procrustes MPJPE, per-frame PA-MPJPE, bone-direction angle error (world and
         after one rotation per take), interior joint-angle error, bone-length ratio/CV, error jitter, time-lag sweep.
         No clipping of errors; unobserved joint samples are excluded and counted.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from g1_imitation.core import adv_eval as ae  # noqa: E402
from g1_imitation.core.s2_triangulation import MultiViewTriangulator  # noqa: E402
from g1_imitation.core.s3_temporal_filter import TemporalFilter  # noqa: E402
from g1_imitation.core.bvh import parse_bvh_and_fk  # noqa: E402

CAMS = ["44310274", "44310277", "44310281", "44310284"]
S1 = ROOT / "outputs" / "advanced_take" / "s1"
RAW = ROOT / "data" / "advanced_take_raw"
OUT = ROOT / "outputs" / "advanced_take"
CAL_TAKES = ["Take_002", "Take_012"]
HELD = ["Take_006", "Take_007", "Take_008"]
FPS = 30.0
C_MIN = 0.35          # pipeline_config.json s2_triangulation.min_2d_confidence
REPROJ_PX = 15.0      # pipeline_config.json s2_triangulation.reprojection_error_threshold_px
MAX_SPEED = 25.0      # m/s, the S2 spec's 0.25 m / 10 ms limit expressed per slot below


def load_cameras(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    cams = d["cameras"]
    P = []
    for c in CAMS:
        if "P" in cams[c]:
            P.append(np.array(cams[c]["P"]))
        else:
            K = np.array(cams[c]["K"])
            R = np.array(cams[c]["R_world_to_camera"])
            t = np.array(cams[c]["t_world_to_camera_m"])
            P.append(K @ np.hstack([R, t[:, None]]))
    return np.array(P), d


def track_take(take, P, use_consensus=True, shifts=None, ref_lag=0):
    d = np.load(S1 / f"{take}_kp27.npz")
    bvh = parse_bvh_and_fk(RAW / "bvh" / f"{take}_Skeleton0.bvh")
    pts = {c: d[f"{c}_pts_s"] for c in CAMS}
    grid, idx = ae.build_slot_grid(pts, bvh["n_motion_frames"], shifts)
    T = len(grid)
    tri = MultiViewTriangulator(c_min=C_MIN, reproj_outlier_threshold_px=REPROJ_PX,
                                max_displacement_m=MAX_SPEED / FPS, use_consensus_search=use_consensus)
    flt = TemporalFilter(freq_hz=FPS, f_c_min=1.2, beta=0.04, f_c_d=1.0, z_nominal_m=0.0)
    raw = np.full((T, 27, 3), np.nan)
    smooth = np.full((T, 27, 3), np.nan)
    cov = np.full((T, 27, 3, 3), np.nan)
    ninl = np.zeros((T, 27), np.int32)
    reproj = np.full((T, 27), np.nan)
    for k in range(T):
        kp = np.zeros((4, 27, 3))
        for ci, c in enumerate(CAMS):
            if idx[c][k] >= 0:
                kp[ci] = d[f"{c}_kp"][idx[c][k]]
        r = tri.process_frame(kp, P)
        raw[k] = r.points_3d_m
        ninl[k] = r.inlier_camera_counts
        reproj[k] = r.reprojection_errors_px
        if r.point_covariances_m2 is not None:
            cov[k] = r.point_covariances_m2
        smooth[k] = flt.process_frame(r.points_3d_m).points_3d_smooth_m
    ref = bvh["motion_world_m"][:, [bvh["name_to_idx"][n] for n in ae.BVH12]]
    rg = grid + int(ref_lag)           # BVH samples read `ref_lag` * 8.33 ms later than the legacy mapping
    ref12 = np.full((T, 12, 3), np.nan)
    okg = (rg >= 0) & (rg < bvh["n_motion_frames"])
    ref12[okg] = ref[rg[okg]]
    return dict(grid=grid, raw=raw, smooth=smooth, cov=cov, ninl=ninl, reproj=reproj, ref_lag=int(ref_lag),
                bvh12=ref12, bvh_all=bvh["motion_world_m"], bvh_idx=bvh["name_to_idx"],
                bvh_frozen=bool(bvh["mocap_frozen"]), n_motion=bvh["n_motion_frames"],
                frame_idx=idx)


def _mpjpe(p, g):
    e = np.linalg.norm(p - g, axis=-1)
    return e[np.isfinite(e)]


def evaluate(tr, key):
    p = tr[key][:, 5:17]
    g = tr["bvh12"]
    e = np.linalg.norm(p - g, axis=-1)                     # (T,12)
    ok = np.isfinite(e)
    out = dict(observed_joint_samples=int(ok.sum()), possible_joint_samples=int(e.size),
               observed_fraction=float(ok.mean()),
               world_mpjpe_mm=float(e[ok].mean() * 1000), world_median_mm=float(np.median(e[ok]) * 1000),
               world_p95_mm=float(np.percentile(e[ok], 95) * 1000),
               per_joint_mpjpe_mm={n: float(np.nanmean(e[:, i]) * 1000) for i, n in enumerate(ae.JOINTS12)})
    full = np.isfinite(p).all((1, 2)) & np.isfinite(g).all((1, 2))
    # take-level similarity and per-frame PA
    if full.sum() > 20:
        s, R, t = ae.umeyama(p[full].reshape(-1, 3), g[full].reshape(-1, 3))
        aligned = (s * (R @ p[full].reshape(-1, 3).T)).T + t
        out["take_procrustes_mpjpe_mm"] = float(np.linalg.norm(aligned - g[full].reshape(-1, 3), axis=1).mean() * 1000)
        out["take_procrustes_scale"] = float(s)
        pa = []
        for a, b in zip(p[full], g[full]):
            s2, R2, t2 = ae.umeyama(a, b)
            pa.append(np.linalg.norm((s2 * (R2 @ a.T)).T + t2 - b, axis=1).mean())
        out["per_frame_pa_mpjpe_mm"] = float(np.mean(pa) * 1000)
        out["full12_frames"] = int(full.sum())
        # segment directions / joint angles on fully observed frames
        dp, dg = ae.segment_directions(p[full]), ae.segment_directions(g[full])
        Rrot = ae.rotation_only_align(np.concatenate([dp[n] for n in dp]), np.concatenate([dg[n] for n in dg]))
        seg_world, seg_rot = {}, {}
        for n in dp:
            seg_world[n] = float(np.mean(ae.angle_between_deg(dp[n], dg[n])))
            seg_rot[n] = float(np.mean(ae.angle_between_deg((Rrot @ dp[n].T).T, dg[n])))
        out["segment_dir_err_deg_world"] = seg_world
        out["segment_dir_err_deg_after_one_rotation"] = seg_rot
        out["segment_dir_err_mean_deg_world"] = float(np.mean([seg_world[n] for n, _, _ in ae.SEGMENTS]))
        out["segment_dir_err_mean_deg_after_rotation"] = float(np.mean([seg_rot[n] for n, _, _ in ae.SEGMENTS]))
        out["global_rotation_residual_deg"] = float(np.degrees(np.arccos(np.clip((np.trace(Rrot) - 1) / 2, -1, 1))))
        ap, ag = ae.interior_angles(p[full]), ae.interior_angles(g[full])
        out["joint_angle_mae_deg"] = {n: float(np.mean(np.abs(ap[n] - ag[n]))) for n in ap}
        out["joint_angle_bias_deg"] = {n: float(np.mean(ap[n] - ag[n])) for n in ap}
        bp, bg = ae.bone_lengths(p[full]), ae.bone_lengths(g[full])
        out["bone_length_ratio_vis_over_bvh"] = {n: float(np.median(bp[n]) / np.median(bg[n])) for n in bp}
        out["bone_length_cv_vision"] = {n: float(np.std(bp[n]) / np.mean(bp[n])) for n in bp}
        out["bone_length_cv_bvh"] = {n: float(np.std(bg[n]) / np.mean(bg[n])) for n in bg}
    out["error_jitter_rms_mm_per_slot"] = ae.jitter_rms(p - g) * 1000
    return out


def lag_sweep(tr, key, ks=range(-48, 49, 2)):
    """MPJPE vs shifting the BVH read index by k BVH samples (8.33 ms each)."""
    p = tr[key][:, 5:17]
    allp = tr["bvh_all"]
    names = [tr["bvh_idx"][n] for n in ae.BVH12]
    res = {}
    for k in ks:
        b = tr["grid"] + k
        v = np.isfinite(tr[key][:, 5:17]).all((1, 2)) & (b >= 0) & (b < tr["n_motion"])
        g = allp[b[v]][:, names]
        e = _mpjpe(p[v], g)
        res[int(k)] = float(e.mean() * 1000) if len(e) else None
    return res


def foot_ground_check(tr):
    """Registration sanity: low percentiles of vision heel/toe/ankle height vs BVH ToeBase/Foot."""
    raw = tr["raw"]
    idx = [15, 16, 17, 18, 19, 20, 21, 22]
    z = raw[:, idx, 2]
    z = z[np.isfinite(z)]
    names = tr["bvh_idx"]
    bz = np.concatenate([tr["bvh_all"][:, names["LeftToeBase"], 2], tr["bvh_all"][:, names["RightToeBase"], 2]])
    return dict(vision_foot_z_p1_m=float(np.percentile(z, 1)), vision_foot_z_p5_m=float(np.percentile(z, 5)),
                bvh_toebase_z_p1_m=float(np.percentile(bz, 1)), bvh_toebase_z_p5_m=float(np.percentile(bz, 5)))


def cache_2d_diff(take):
    """Fresh S1 vs legacy cached 2D on the 12 body joints (px), same frames/PTS."""
    p = ROOT / "outputs" / "advanced_take_inspection" / "kps2d_cache" / f"{take}_2d.pkl"
    if not p.exists():
        return None
    cache = pickle.loads(p.read_bytes())  # trusted local cache produced by the earlier workflow phase
    d = np.load(S1 / f"{take}_kp27.npz")
    diffs = []
    for c in CAMS:
        rec = cache[c]
        k = d[f"{c}_kp"]
        if len(rec) != len(k):
            return dict(status="frame_count_mismatch")
        for i, r in enumerate(rec):
            if r["kps"] is None:
                continue
            m = (np.asarray(r["scores"][5:17]) >= 0.5) & (k[i, 5:17, 2] >= 0.5)
            if m.any():
                diffs.append(np.linalg.norm(np.asarray(r["kps"][5:17])[m] - k[i, 5:17, :2][m], axis=1))
    diffs = np.concatenate(diffs)
    return dict(median_px=float(np.median(diffs)), mean_px=float(diffs.mean()), p95_px=float(np.percentile(diffs, 95)),
                n=int(len(diffs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", default=str(OUT / "cameras_selfcal.json"))
    ap.add_argument("--tag", default="selfcal")
    ap.add_argument("--takes", nargs="+", default=CAL_TAKES + HELD)
    ap.add_argument("--lag", type=int, default=0, help="read BVH `lag` samples (8.33 ms) later; diagnostic")
    ap.add_argument("--literal-s2", action="store_true", help="use the literal two-round residual rejection")
    ap.add_argument("--shift", nargs="*", default=[], help="cam=slots diagnostic shift, e.g. 44310284=-1")
    args = ap.parse_args()
    shifts = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in args.shift}
    P, camjson = load_cameras(Path(args.cameras))
    odir = OUT / "s2s3" / args.tag
    odir.mkdir(parents=True, exist_ok=True)
    metrics = dict(tag=args.tag, cameras=args.cameras, s2_consensus=not args.literal_s2, c_min=C_MIN,
                   reproj_threshold_px=REPROJ_PX, slot_shifts=shifts, ref_lag_bvh_samples=args.lag, takes={})
    for take in args.takes:
        tr = track_take(take, P, use_consensus=not args.literal_s2, shifts=shifts or None, ref_lag=args.lag)
        if tr["bvh_frozen"]:
            raise RuntimeError(f"{take}: BVH frozen, no reference")
        role = "calibration_fit_not_independent" if take in CAL_TAKES else "held_out"
        m = dict(role=role, slots=int(len(tr["grid"])),
                 raw=evaluate(tr, "raw"), filtered=evaluate(tr, "smooth"),
                 lag_raw=lag_sweep(tr, "raw"), lag_filtered=lag_sweep(tr, "smooth"),
                 foot_ground=foot_ground_check(tr), cache_2d_diff=cache_2d_diff(take),
                 mean_inlier_reproj_px=float(np.nanmean(tr["reproj"][:, 5:17])),
                 frames_with_ge2_views=float(np.mean((np.isfinite(tr["raw"][:, 5:17]).all((1, 2))))))
        metrics["takes"][take] = m
        np.savez_compressed(odir / f"{take}.npz", grid=tr["grid"], raw=tr["raw"], smooth=tr["smooth"], cov=tr["cov"],
                            bvh12=tr["bvh12"], ninl=tr["ninl"], reproj=tr["reproj"])
        print(take, role, "raw MPJPE %.1f mm | procrustes %.1f | PA %.1f | seg-dir %.1f deg (rot-aligned %.1f) | obs %.3f" % (
            m["raw"]["world_mpjpe_mm"], m["raw"].get("take_procrustes_mpjpe_mm", np.nan),
            m["raw"].get("per_frame_pa_mpjpe_mm", np.nan), m["raw"].get("segment_dir_err_mean_deg_world", np.nan),
            m["raw"].get("segment_dir_err_mean_deg_after_rotation", np.nan), m["raw"]["observed_fraction"]), flush=True)
    (odir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
