"""RGB-only world registration of the self-calibrated AdvancedTake rig (no BVH anywhere in the result).

Inputs : outputs/advanced_take/cameras_selfcal.json  (only K and the RELATIVE camera poses are used; they
         come from the BVH-free bundle adjustment. The BVH similarity stored there is removed: every camera is first
         re-expressed in the cam0 frame with the cam0-cam1 baseline normalised to 1, so nothing of it survives.)
         s1/<Take>_kp27.npz of the calibration takes (Take_002, Take_012) - RGB detections only.
Output : outputs/advanced_take/cameras_rgbonly.json  (K, R, t, P in an RGB-derived Z-up metric world)
Method : 1. triangulate the 27 points with S2 in the gauge frame;
         2. up = mean pelvis->neck direction (initial guess);
         3. ground = RANSAC plane through the per-slot lowest heel/toe point, refined by SVD; z = plane normal;
         4. metric scale: median pelvis height above the ground on straight-leg slots := population prior
            `world_registration.pelvis_height_prior_m` (0.92 m). This is an ASSUMPTION about the subject, not a
            measurement; the retargeter (GMR) rescales by its own human-height estimate, so the G1 joint reference is insensitive
            to it, but root translation and S2/S3 metre thresholds are not;
         5. origin = ground projection of the mean pelvis, x = cam0 optical axis projected onto the ground.
Failure: raises RuntimeError when too few ground candidates/inliers exist (no fallback to any reference).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from g1_imitation.core.constants import load_config  # noqa: E402
from g1_imitation.core.s2_triangulation import MultiViewTriangulator  # noqa: E402

CAMS = ["44310274", "44310277", "44310281", "44310284"]
OUT = ROOT / "outputs" / "advanced_take"
CAL_TAKES = ["Take_002", "Take_012"]
FOOT = [17, 18, 19, 20, 21, 22]          # big toe, small toe, heel (L, R)
L_HIP, R_HIP, L_SH, R_SH = 11, 12, 5, 6
L_KNEE, R_KNEE, L_ANK, R_ANK = 13, 14, 15, 16


def gauge_cameras(cams: dict):
    """Re-express cameras in the cam0 frame with |C1 - C0| = 1. Returns K list and (R, t) list."""
    K = [np.array(cams[c]["K"]) for c in CAMS]
    Rw = [np.array(cams[c]["R_world_to_camera"]) for c in CAMS]
    tw = [np.array(cams[c]["t_world_to_camera_m"]) for c in CAMS]
    R0, t0 = Rw[0], tw[0]
    # X_world = R0^T (X_c0 - t0)  ->  camera c: R_c R0^T X_c0 + (t_c - R_c R0^T t0)
    Rg = [R @ R0.T for R in Rw]
    tg = [t - R @ R0.T @ t0 for R, t in zip(Rw, tw)]
    base = np.linalg.norm(-Rg[1].T @ tg[1])          # cam1 centre in the cam0 frame
    tg = [t / base for t in tg]
    return K, Rg, tg


def projections(K, R, t):
    return np.array([k @ np.hstack([r, tt[:, None]]) for k, r, tt in zip(K, R, t)])


def triangulate_takes(P):
    tri = MultiViewTriangulator(c_min=0.35, reproj_outlier_threshold_px=15.0, max_displacement_m=1e9,
                                use_consensus_search=True)
    out = []
    for take in CAL_TAKES:
        d = np.load(OUT / "s1" / f"{take}_kp27.npz")
        n = min(len(d[f"{c}_kp"]) for c in CAMS)
        # end alignment between cameras (same hypothesis as S2 evaluation; no BVH involved)
        idx = {c: np.arange(len(d[f"{c}_kp"]) - n, len(d[f"{c}_kp"])) for c in CAMS}
        for k in range(n):
            kp = np.stack([d[f"{c}_kp"][idx[c][k]] for c in CAMS])
            out.append(tri.process_frame(kp, P).points_3d_m)
    return np.array(out)


def ransac_plane(X, thr, iters=4000, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        a, b, c = X[rng.choice(len(X), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((X - a) @ n) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    c = X[best].mean(0)
    n = np.linalg.svd(X[best] - c)[2][-1]
    return n, c, best


def knee_angle(hip, knee, ank):
    a, b = hip - knee, ank - knee
    cos = np.sum(a * b, -1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def main():
    sel = json.loads((OUT / "cameras_selfcal.json").read_text(encoding="utf-8"))
    K, Rg, tg = gauge_cameras(sel["cameras"])
    X = triangulate_takes(projections(K, Rg, tg))                       # (N,27,3) gauge units
    pelvis = 0.5 * (X[:, L_HIP] + X[:, R_HIP])
    neck = 0.5 * (X[:, L_SH] + X[:, R_SH])
    spine = neck - pelvis
    ok = np.isfinite(spine).all(1)
    up0 = spine[ok].mean(0)
    up0 /= np.linalg.norm(up0)
    leg = np.nanmedian(np.linalg.norm(X[:, L_HIP] - X[:, L_KNEE], axis=1) + np.linalg.norm(X[:, L_KNEE] - X[:, L_ANK], axis=1))

    up = up0
    for _ in range(2):                                                   # re-pick lowest points with refined up
        feet = X[:, FOOT]
        h = feet @ up
        h[~np.isfinite(h)] = np.inf
        j = np.argmin(h, axis=1)
        cand = feet[np.arange(len(X)), j]
        cand = cand[np.isfinite(cand).all(1)]
        if len(cand) < 100:
            raise RuntimeError("too few ground candidates")
        n, c0, inl = ransac_plane(cand, thr=0.03 * leg / 0.85)
        if inl.sum() < 50:
            raise RuntimeError("too few ground inliers")
        up = n if n @ up0 > 0 else -n
    tilt_vs_spine = float(np.degrees(np.arccos(np.clip(up @ up0, -1, 1))))

    straight = (knee_angle(X[:, L_HIP], X[:, L_KNEE], X[:, L_ANK]) > 165) & (knee_angle(X[:, R_HIP], X[:, R_KNEE], X[:, R_ANK]) > 165)
    h_pel = (pelvis - c0) @ up
    hs = h_pel[straight & np.isfinite(h_pel)]
    if len(hs) < 50:
        raise RuntimeError("too few straight-leg frames for the scale prior")
    prior = float(load_config()["world_registration"]["pelvis_height_prior_m"])
    scale = prior / float(np.median(hs))

    # world axes: z = up, x = cam0 optical axis projected on the ground, origin under mean pelvis
    z = up
    x = np.array([0.0, 0.0, 1.0]) - (np.array([0.0, 0.0, 1.0]) @ z) * z         # cam0 optical axis in gauge frame
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    Rwg = np.stack([x, y, z])                                                    # gauge -> world rotation
    pm = np.nanmean(pelvis[np.isfinite(pelvis).all(1)], axis=0)
    origin = pm - ((pm - c0) @ z) * z                                            # on the plane
    # X_w = scale * Rwg (X_g - origin)  ->  camera: R_g X_g + t_g = R_g (Rwg^T X_w / scale + origin) + t_g
    cams_out = {}
    for ci, cam in enumerate(CAMS):
        Rw = Rg[ci] @ Rwg.T
        tw = scale * (Rg[ci] @ origin + tg[ci])
        cams_out[cam] = dict(K=K[ci].tolist(), R_world_to_camera=Rw.tolist(), t_world_to_camera_m=tw.tolist(),
                             P=(K[ci] @ np.hstack([Rw, tw[:, None]])).tolist(),
                             width=sel["cameras"][cam]["width"], height=sel["cameras"][cam]["height"])
    ground_resid_mm = np.abs((cand[inl] - c0) @ up) * scale * 1000
    payload = dict(
        provenance="RGB-only: BVH-free BA intrinsics/relative poses (cameras_selfcal.json, BVH similarity removed); "
                   "gravity/ground from RANSAC on lowest triangulated heel/toe points; metric scale from a pelvis-height "
                   "prior; calibration takes only",
        calibration_takes=CAL_TAKES, focal_px=sel["focal_px"], ba=sel["ba"],
        registration=dict(n_slots=int(len(X)), ground_candidates=int(len(cand)), ground_inliers=int(inl.sum()),
                          ground_resid_median_mm=float(np.median(ground_resid_mm)),
                          ground_resid_p95_mm=float(np.percentile(ground_resid_mm, 95)),
                          up_vs_mean_spine_deg=tilt_vs_spine, straight_leg_slots=int(len(hs)),
                          pelvis_height_prior_m=prior, pelvis_height_gauge_median=float(np.median(hs)),
                          pelvis_height_gauge_iqr=[float(np.percentile(hs, 25)), float(np.percentile(hs, 75))],
                          metres_per_gauge_unit=float(scale)),
        cameras=cams_out,
        limitations=["metric scale is a population prior (pelvis height 0.92 m), not measured",
                     "world yaw/origin are conventions (cam0 axis, mean pelvis)",
                     "inter-camera sync is the unverified end-alignment hypothesis", "no distortion model"])
    (OUT / "cameras_rgbonly.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["registration"], indent=1))


if __name__ == "__main__":
    main()
