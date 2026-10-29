"""BVH-free self-calibration of the 4 AdvancedTake cameras from fresh S1 2D keypoints.

Inputs : outputs/advanced_take/s1/<Take>_kp27.npz (calibration takes only).
Model  : pinhole, fx = fy = f_c (free per camera), principal point = image centre, zero distortion.
Method : essential-matrix init (cam 0 reference) -> PnP for the other cameras -> robust sparse bundle
         adjustment over 2D reprojection only. Metric scale, orientation and origin are then fixed by ONE
         similarity transform (7 parameters) fitted on the calibration takes against the BVH template-aligned
         world. No BVH joint enters the camera intrinsics/extrinsics and no held-out take enters anything.
Output : outputs/advanced_take/cameras_selfcal.json (K, R_w2c, t_w2c, P) + diagnostics.
Time   : inter-camera slot offsets are only DIAGNOSED (epipolar consistency vs integer slot shift);
         the default offset stays 0, i.e. the legacy end-alignment hypothesis, unverified.
Failure: <200 correspondences, failed pose recovery or non-convergence abort; nothing is written.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from g1_imitation.core.adv_eval import build_slot_grid, robust_umeyama, BVH12  # noqa: E402
from g1_imitation.core.bvh import parse_bvh_and_fk  # noqa: E402

CAMS = ["44310274", "44310277", "44310281", "44310284"]
S1 = ROOT / "outputs" / "advanced_take" / "s1"
RAW = ROOT / "data" / "advanced_take_raw"
OUT = ROOT / "outputs" / "advanced_take"
CAL_TAKES = ["Take_002", "Take_012"]
SCORE_MIN = 0.5
BODY = list(range(5, 17))
W, H = 2048, 1544


def K_of(f):
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])


def load_take(take, shifts=None):
    d = np.load(S1 / f"{take}_kp27.npz")
    bvh = parse_bvh_and_fk(RAW / "bvh" / f"{take}_Skeleton0.bvh")
    pts = {c: d[f"{c}_pts_s"] for c in CAMS}
    grid, idx = build_slot_grid(pts, bvh["n_motion_frames"], shifts)
    kp = np.full((len(grid), 4, 27, 3), np.nan)
    for ci, c in enumerate(CAMS):
        ok = idx[c] >= 0
        kp[ok, ci] = d[f"{c}_kp"][idx[c][ok]]
    return grid, kp, bvh


def observations(kps, stride=1):
    """list of (slot_key, joint, cam, u, v) for confident body joints."""
    obs = []
    for k, kp in enumerate(kps):
        if k % stride:
            continue
        for j in BODY:
            for c in range(4):
                u, v, s = kp[c, j]
                if np.isfinite(s) and s >= SCORE_MIN:
                    obs.append((k, j, c, u, v))
    return obs


def sync_diagnostic(kp, ref=0, shifts=range(-3, 4)):
    """Median symmetric epipolar distance (px) of cam `ref`-vs-cam c vs integer slot shift of c."""
    res = {}
    for c in range(4):
        if c == ref:
            continue
        row = {}
        for s in shifts:
            a, b = [], []
            for k in range(len(kp)):
                k2 = k + s
                if not (0 <= k2 < len(kp)):
                    continue
                for j in BODY:
                    if kp[k, ref, j, 2] >= SCORE_MIN and kp[k2, c, j, 2] >= SCORE_MIN:
                        a.append(kp[k, ref, j, :2])
                        b.append(kp[k2, c, j, :2])
            a, b = np.array(a), np.array(b)
            if len(a) < 200:
                row[s] = None
                continue
            F, m = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 3.0, 0.999)
            if F is None:
                row[s] = None
                continue
            m = m.ravel().astype(bool)
            ha = np.c_[a, np.ones(len(a))]
            hb = np.c_[b, np.ones(len(b))]
            l2 = ha @ F.T
            l1 = hb @ F
            d = np.abs(np.sum(hb * l2, 1)) / np.hypot(l2[:, 0], l2[:, 1])
            d += np.abs(np.sum(ha * l1, 1)) / np.hypot(l1[:, 0], l1[:, 1])
            row[s] = dict(n=int(len(a)), inlier_frac=float(m.mean()), median_sym_epi_px=float(np.median(d / 2)))
        res[CAMS[c]] = row
    return res


def triangulate_pair(P0, P1, a, b):
    X = cv2.triangulatePoints(P0, P1, a.T, b.T)
    return (X[:3] / X[3]).T


def cost_with_f(f, pair_pts):
    K = K_of(f)
    a, b = pair_pts
    E, m = cv2.findEssentialMat(a, b, K, cv2.RANSAC, 0.999, 1.5)
    if E is None:
        return np.inf, None
    n, R, t, mask = cv2.recoverPose(E, a, b, K, mask=m)
    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K @ np.hstack([R, t])
    inl = mask.ravel() > 0
    X = triangulate_pair(P0, P1, a[inl], b[inl])
    pr = (P1 @ np.c_[X, np.ones(len(X))].T).T
    pr = pr[:, :2] / pr[:, 2:]
    return float(np.median(np.linalg.norm(pr - b[inl], axis=1))), (R, t.ravel(), n)


def run_ba(obs_pack, cams0, f0, n_pts, X0):
    """Sparse BA. cams0: list[(rvec,t)] for cams 1..3; f0: 4 focal lengths; X0: (n_pts,3)."""
    pt_i, cam_i, uv = obs_pack
    nobs = len(pt_i)

    def unpack(v):
        f = v[:4]
        cams = [(v[4 + 6 * i: 7 + 6 * i], v[7 + 6 * i: 10 + 6 * i]) for i in range(3)]
        X = v[22:].reshape(-1, 3)
        return f, cams, X

    def resid(v):
        f, cams, X = unpack(v)
        r = np.empty((nobs, 2))
        for c in range(4):
            m = cam_i == c
            if c == 0:
                Xc = X[pt_i[m]]
            else:
                rv, t = cams[c - 1]
                t = t / np.linalg.norm(cams[0][1])  # fix gauge scale: |t_cam1| == 1
                Xc = (Rotation.from_rotvec(rv).as_matrix() @ X[pt_i[m]].T).T + t
            z = np.maximum(Xc[:, 2], 1e-6)
            r[m, 0] = f[c] * Xc[:, 0] / z + W / 2 - uv[m, 0]
            r[m, 1] = f[c] * Xc[:, 1] / z + H / 2 - uv[m, 1]
        return r.ravel()

    v0 = np.r_[f0, np.concatenate([np.r_[rv, t] for rv, t in cams0]), X0.ravel()]
    A = lil_matrix((2 * nobs, len(v0)), dtype=int)
    rows = np.arange(nobs)
    for c in range(4):
        m = np.flatnonzero(cam_i == c)
        for k in (0, 1):
            A[2 * m + k, c] = 1
        if c > 0:
            for k in (0, 1):
                for q in range(6):
                    A[2 * m + k, 4 + 6 * (c - 1) + q] = 1
                if c == 1 or True:
                    pass
    # cam1 translation norm couples cams 2,3 through the gauge normalisation
    for c in (2, 3):
        m = np.flatnonzero(cam_i == c)
        for k in (0, 1):
            for q in range(3, 6):
                A[2 * m + k, 4 + q] = 1
    for k in (0, 1):
        for q in range(3):
            A[2 * rows + k, 22 + 3 * pt_i + q] = 1
    lo = np.full(len(v0), -np.inf)
    hi = np.full(len(v0), np.inf)
    lo[:4], hi[:4] = 600.0, 6000.0
    r_init = resid(v0).reshape(-1, 2)
    e_init = np.linalg.norm(r_init, axis=1)
    print("BA init reproj median px per cam", [round(float(np.median(e_init[cam_i == c])), 1) for c in range(4)], flush=True)
    sol = least_squares(resid, v0, jac_sparsity=A, loss="soft_l1", f_scale=3.0, method="trf",
                        x_scale="jac", max_nfev=80, bounds=(lo, hi), verbose=0)
    f, cams, X = unpack(sol.x)
    t1n = np.linalg.norm(cams[0][1])
    cams = [(rv, t / t1n) for rv, t in cams]
    r = resid(sol.x).reshape(-1, 2)
    return f, cams, X, np.linalg.norm(r, axis=1), sol


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    takes = {t: load_take(t) for t in CAL_TAKES}
    diag = {t: sync_diagnostic(takes[t][1]) for t in CAL_TAKES}

    # ---- correspondences cam0-cam1 for focal search -------------------------------------------------
    a, b = [], []
    for t in CAL_TAKES:
        kp = takes[t][1]
        m = (kp[:, 0, BODY, 2] >= SCORE_MIN) & (kp[:, 1, BODY, 2] >= SCORE_MIN)
        a.append(kp[:, 0, BODY, :2][m])
        b.append(kp[:, 1, BODY, :2][m])
    a, b = np.concatenate(a), np.concatenate(b)
    if len(a) < 200:
        raise RuntimeError("too few cam0-cam1 correspondences")
    scan = {f: cost_with_f(f, (a, b))[0] for f in (900, 1200, 1500, 1800, 2200, 2800, 3500)}
    f0 = min(scan, key=scan.get)
    _, (R1, t1, _) = cost_with_f(f0, (a, b))

    # ---- build shared point set (slot,joint) with >=2 views ------------------------------------------
    pts, obs_pt, obs_cam, obs_uv = {}, [], [], []
    for ti, t in enumerate(CAL_TAKES):
        kp = takes[t][1]
        for k in range(0, len(kp), 3):
            for j in BODY:
                vis = [c for c in range(4) if kp[k, c, j, 2] >= SCORE_MIN]
                if len(vis) < 2:
                    continue
                key = (ti, k, j)
                pts[key] = len(pts)
                for c in vis:
                    obs_pt.append(pts[key])
                    obs_cam.append(c)
                    obs_uv.append(kp[k, c, j, :2])
    obs_pt, obs_cam, obs_uv = np.array(obs_pt), np.array(obs_cam), np.array(obs_uv)

    # ---- initial 3D from cam0/cam1, PnP for cams 2/3 ---------------------------------------------------
    K = K_of(f0)
    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K @ np.hstack([R1, t1[:, None]])
    X0 = np.full((len(pts), 3), np.nan)
    by_pt = {}
    for p, c, uv in zip(obs_pt, obs_cam, obs_uv):
        by_pt.setdefault(p, {})[c] = uv
    for p, d in by_pt.items():
        if 0 in d and 1 in d:
            X0[p] = triangulate_pair(P0, P1, d[0][None], d[1][None])[0]
    cams0 = [(cv2.Rodrigues(R1)[0].ravel(), t1)]
    for c in (2, 3):
        pp = [(X0[p], d[c]) for p, d in by_pt.items() if c in d and np.isfinite(X0[p]).all()]
        if len(pp) < 100:
            raise RuntimeError(f"PnP init: too few points for cam {c}")
        ok, rv, tv, _ = cv2.solvePnPRansac(np.array([x for x, _ in pp]), np.array([u for _, u in pp]), K, None,
                                           iterationsCount=400, reprojectionError=6.0, flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            raise RuntimeError(f"PnP failed for cam {c}")
        cams0.append((rv.ravel(), tv.ravel()))
    # initial DLT for points lacking cam0/cam1 pair
    Ps = [P0, P1] + [K @ np.hstack([cv2.Rodrigues(rv)[0], t[:, None]]) for rv, t in cams0[1:]]
    for p, d in by_pt.items():
        if np.isfinite(X0[p]).all():
            continue
        rows = []
        for c, uv in d.items():
            rows += [uv[0] * Ps[c][2] - Ps[c][0], uv[1] * Ps[c][2] - Ps[c][1]]
        _, _, Vt = np.linalg.svd(np.array(rows))
        X0[p] = Vt[-1, :3] / Vt[-1, 3]
    good = np.isfinite(X0).all(1)
    keep_obs = good[obs_pt]
    remap = -np.ones(len(pts), int)
    remap[good] = np.arange(good.sum())
    pack = (remap[obs_pt[keep_obs]], obs_cam[keep_obs], obs_uv[keep_obs])
    X0g = X0[good]
    # normalise gauge: cam1 translation unit norm
    s = np.linalg.norm(cams0[0][1])
    cams0 = [(rv, t / s) for rv, t in cams0]
    X0g = X0g / s

    f_a, cams_a, X_a, err_a, sol = run_ba(pack, cams0, np.full(4, f0), good.sum(), X0g)
    # second pass: drop gross outliers (>4x median) and re-solve
    thr = max(6.0, 4 * float(np.median(err_a)))
    k2 = err_a < thr
    used_pts = np.unique(pack[0][k2])
    remap2 = -np.ones(good.sum(), int)
    remap2[used_pts] = np.arange(len(used_pts))
    pack2 = (remap2[pack[0][k2]], pack[1][k2], pack[2][k2])
    f_b, cams_b, X_b, err_b, sol2 = run_ba(pack2, cams_a, f_a, len(used_pts), X_a[used_pts])

    # ---- cameras in the arbitrary-scale cam0 frame ---------------------------------------------------
    Rt = [(np.eye(3), np.zeros(3))] + [(Rotation.from_rotvec(rv).as_matrix(), t) for rv, t in cams_b]

    # ---- one 7-DoF similarity to the BVH world, fitted on calibration takes ----------------------------
    src, dst = [], []
    for t in CAL_TAKES:
        grid, kp, bvh = takes[t]
        for k in range(0, len(kp), 2):
            for ji, j in enumerate(BODY):
                vis = [c for c in range(4) if kp[k, c, j, 2] >= SCORE_MIN]
                if len(vis) < 3:
                    continue
                rows = []
                for c in vis:
                    Kc = K_of(f_b[c])
                    Pc = Kc @ np.hstack([Rt[c][0], Rt[c][1][:, None]])
                    uv = kp[k, c, j, :2]
                    rows += [uv[0] * Pc[2] - Pc[0], uv[1] * Pc[2] - Pc[1]]
                _, _, Vt = np.linalg.svd(np.array(rows))
                X = Vt[-1, :3] / Vt[-1, 3]
                src.append(X)
                dst.append(bvh["motion_world_m"][grid[k], bvh["name_to_idx"][BVH12[ji]]])
    src, dst = np.array(src), np.array(dst)
    sc, Ra, ta = robust_umeyama(src, dst)
    cams_out = {}
    for ci, c in enumerate(CAMS):
        Rw = Rt[ci][0] @ Ra.T
        tw = sc * Rt[ci][1] - Rw @ ta
        Kc = K_of(f_b[ci])
        cams_out[c] = dict(K=Kc.tolist(), R_world_to_camera=Rw.tolist(), t_world_to_camera_m=tw.tolist(),
                           P=(Kc @ np.hstack([Rw, tw[:, None]])).tolist(), width=W, height=H)
    res_world = np.linalg.norm((sc * (Ra @ src.T)).T + ta - dst, axis=1)
    payload = dict(
        provenance="BVH-free pinhole self-calibration from fresh S1 2D (calibration takes only); "
                   "7-DoF similarity to BVH world fitted on the same calibration takes",
        calibration_takes=CAL_TAKES, model="fx=fy free per camera, pp=image centre, no distortion",
        focal_scan_median_px=scan, focal_init=f0, focal_px=[float(x) for x in f_b],
        ba=dict(n_points=int(len(used_pts)), n_obs=int(k2.sum()),
                reproj_median_px=float(np.median(err_b)), reproj_mean_px=float(np.mean(err_b)),
                reproj_p95_px=float(np.percentile(err_b, 95)), first_pass_median_px=float(np.median(err_a))),
        similarity_to_bvh_world=dict(scale=float(sc), n_pairs=int(len(src)),
                                     fit_resid_median_mm=float(np.median(res_world) * 1000),
                                     fit_resid_mean_mm=float(np.mean(res_world) * 1000)),
        sync_diagnostic_epipolar=diag, slot_shift_applied=0,
        cameras=cams_out,
        limitations=["no distortion model", "metric scale/orientation come from the BVH world alignment, "
                     "not from a measured calibration object", "sync is the unverified end-alignment hypothesis"])
    (OUT / "cameras_selfcal.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("focal_init", "focal_px", "ba", "similarity_to_bvh_world")}, indent=1))
    for t in CAL_TAKES:
        for c, row in diag[t].items():
            best = min((s for s in row if row[s]), key=lambda s: row[s]["median_sym_epi_px"])
            print(t, c, "best_shift", best, {s: (round(v["median_sym_epi_px"], 2) if v else None) for s, v in row.items()})


if __name__ == "__main__":
    main()
