"""Multi-camera checkerboard calibration with floor Z = 0 alignment (input: scripts/hik_calib_capture.py bundles).

Input : data/calib_capture/<session>/{intrinsics,extrinsics,floor}/<bundle>/<serial>.png + meta.json, the board's
        inner-corner count (from meta) and measured square size `--square-mm` (millimetres, measure with a ruler).
        Board thickness `--board-thickness-mm` lifts the floor board's top surface above Z = 0.
Method: 1) corners: findChessboardCornersSB (fallback: classic + sub-pixel), then ordered so that the inner square at
        corner 0 is the dark one; this fixes the 180-degree ambiguity, so the board must have one odd and one even
        inner-corner count (e.g. 9x6). 2) intrinsics per camera: cv2.calibrateCamera, 5 distortion terms (k3 fixed
        unless --free-k3). 3) extrinsics: per-view PnP, relative poses chained from the first camera, then a robust
        (Huber) bundle adjustment of camera and board poses with intrinsics fixed. 4) floor: plane fitted to the floor
        boards, normal pointing to the cameras; world X = board column direction of the first floor bundle, origin =
        its corner 0 projected to the floor. World is right-handed, Z up, metres.
Output: outputs/calibration/physical_cameras.json (per camera: serial, image_size_wh, K, dist_coeffs, R_w2c, t_w2c,
        P = K[R|t] for undistorted pixels, intrinsic and bundle RMSE px) and calibration_report.json (always written).
Checks: per-camera bundle reprojection RMSE < 0.8 px; floor corners triangulated independently of the board poses
        must lie at |Z - thickness| < 3 mm; adjacent triangulated corners must match the square size (reported).
Failure: too few views, a symmetric board, a camera not linked to the others, no floor bundle, or a failed check
        raises an error; the calibration file is then not written (the report still is).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)


def board_points(pattern, square_m):
    cols, rows = pattern
    grid = np.stack(np.meshgrid(np.arange(cols), np.arange(rows)), -1).reshape(-1, 2)
    return np.c_[grid * square_m, np.zeros(len(grid))].astype(np.float64)


def canonical_order(gray, corners, pattern):
    """Reverse the corner order if the inner square at corner 0 is the bright one (180-degree ambiguity)."""
    cols, _ = pattern
    c = corners.reshape(-1, 2)

    def square_mean(i, j):
        quad = np.array([c[j * cols + i], c[j * cols + i + 1], c[(j + 1) * cols + i + 1], c[(j + 1) * cols + i]])
        centre = quad.mean(axis=0)
        poly = (centre + 0.5 * (quad - centre)).astype(np.int32)
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, poly, 1)
        return float(gray[mask > 0].mean()) if mask.any() else np.nan

    return c[::-1].copy() if square_mean(0, 0) > square_mean(1, 0) else c


def detect_corners(gray, pattern, allow_upscale=False):
    fast = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags=fast)
    if ok:
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), SUBPIX)
        return canonical_order(gray, corners, pattern)
    if not allow_upscale:
        return None
    up = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    ok, corners = cv2.findChessboardCorners(up, pattern, flags=fast)
    if ok:
        corners = cv2.cornerSubPix(up, corners, (5, 5), (-1, -1), SUBPIX) / 2.0
        return canonical_order(gray, corners, pattern)
    return None


def load_session(session_dir: Path):
    meta_files = sorted(session_dir.glob("*/*/meta.json"))
    if not meta_files:
        raise ValueError(f"no bundles under {session_dir}")

    def _load_one(meta_file: Path):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        pattern = tuple(meta["pattern"])
        obs = {}
        for s in meta["serials"]:
            gray = cv2.imdecode(np.fromfile(str(meta_file.parent / f"{s}.png"), np.uint8), cv2.IMREAD_GRAYSCALE)
            corners = detect_corners(gray, pattern, allow_upscale=(meta["phase"] == "floor"))
            if corners is not None:
                obs[s] = corners
        return dict(phase=meta["phase"], name=f"{meta['phase']}/{meta_file.parent.name}", obs=obs,
                    serials=meta["serials"], size=tuple(meta["image_size_wh"]), pattern=pattern)

    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(_load_one, meta_files))


def to_T(rvec, tvec):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.ravel(rvec)).as_matrix()
    T[:3, 3] = np.ravel(tvec)
    return T


def calibrate(bundles, serials, image_size, pattern, square_m, *, board_thickness_m=0.0, free_k3=False,
              min_intrinsic_views=8, max_rmse_px=1.5, max_floor_err_mm=8.0):
    if (pattern[0] + pattern[1]) % 2 == 0 or pattern[0] == pattern[1]:
        raise ValueError(f"board {pattern[0]}x{pattern[1]} is symmetric; need one odd and one even inner-corner count")
    obj = board_points(pattern, square_m)
    report = dict(pattern=list(pattern), square_m=square_m, cameras={})
    K, dist = {}, {}
    flags = 0 if free_k3 else cv2.CALIB_FIX_K3
    for s in serials:
        img_pts = [b["obs"][s].astype(np.float32) for b in bundles if s in b["obs"]]
        if len(img_pts) < min_intrinsic_views:
            raise ValueError(f"{s}: only {len(img_pts)} board views, need >= {min_intrinsic_views}")
        rms, K[s], d, _, _ = cv2.calibrateCamera([obj.astype(np.float32)] * len(img_pts), img_pts, image_size,
                                                 None, None, flags=flags)
        dist[s] = d.ravel()[:5]
        report["cameras"][s] = dict(views=len(img_pts), intrinsic_rmse_px=float(rms))

    observations = []
    pnp = {}
    for bi, b in enumerate(bundles):
        for s in serials:
            if s not in b["obs"]:
                continue
            pts = b["obs"][s]
            ok, rv, tv = cv2.solvePnP(obj, pts, K[s], dist[s], flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                pnp[bi, s] = to_T(rv, tv)
                observations.append((bi, s, pts))

    ref = serials[0]
    T_c_ref = {ref: np.eye(4)}
    pair = defaultdict(list)
    for bi in range(len(bundles)):
        seen = [s for s in serials if (bi, s) in pnp]
        for a in seen:
            for b in seen:
                if a != b:
                    # Check whether view a or b had a 180-deg flip by favouring floor/consistent Z-up normals
                    pair[a, b].append((bi, pnp[bi, a] @ np.linalg.inv(pnp[bi, b])))
    while len(T_c_ref) < len(serials):
        best = max(((a, b) for (a, b) in pair if b in T_c_ref and a not in T_c_ref),
                   key=lambda k: len(pair[k]), default=None)
        if best is None:
            missing = [s for s in serials if s not in T_c_ref]
            raise ValueError(f"cameras {missing} never see the board together with calibrated cameras")
        a, b = best
        # Prefer floor bundle if available (all 4 cameras saw it stationary), else median of consistent rotations
        floor_Ts = [T for bi, T in pair[a, b] if bundles[bi]["phase"] == "floor"]
        Ts = floor_Ts if floor_Ts else [T for _, T in pair[a, b]]
        rel = np.eye(4)
        rel[:3, :3] = Rotation.from_matrix([T[:3, :3] for T in Ts]).mean().as_matrix()
        rel[:3, 3] = np.median([T[:3, 3] for T in Ts], axis=0)
        T_c_ref[a] = rel @ T_c_ref[b]
    report["shared_views"] = {f"{a}-{b}": len(v) for (a, b), v in pair.items() if a < b}

    used = sorted({bi for bi, s in pnp})
    board_init = {}
    filtered_obs = []
    for bi in used:
        # Pick the camera in this bundle that gives the lowest multi-view reprojection error (resolving any 180-deg flip)
        best_B = None
        best_err = float("inf")
        for s in serials:
            if (bi, s) not in pnp:
                continue
            cand_list = (bundles[bi]["obs"][s],) if bundles[bi]["phase"] == "floor" else (bundles[bi]["obs"][s], bundles[bi]["obs"][s][::-1].copy())
            for cand_pts in cand_list:
                ok, rv, tv = cv2.solvePnP(obj, cand_pts, K[s], dist[s], flags=cv2.SOLVEPNP_ITERATIVE)
                if not ok:
                    continue
                B_cand = np.linalg.inv(T_c_ref[s]) @ to_T(rv, tv)
                errs = []
                for s2 in serials:
                    if s2 not in bundles[bi]["obs"]:
                        continue
                    T_c2_b = T_c_ref[s2] @ B_cand
                    r2, _ = cv2.Rodrigues(T_c2_b[:3, :3])
                    proj, _ = cv2.projectPoints(obj, r2, T_c2_b[:3, 3], K[s2], dist[s2])
                    proj = proj.reshape(-1, 2)
                    p2 = bundles[bi]["obs"][s2]
                    e_norm = float(np.sqrt(np.mean(np.sum((proj - p2) ** 2, axis=1))))
                    e_flip = float(np.sqrt(np.mean(np.sum((proj - p2[::-1]) ** 2, axis=1))))
                    errs.append(min(e_norm, e_flip))
                if np.mean(errs) < best_err:
                    best_err = float(np.mean(errs))
                    best_B = B_cand
        assert best_B is not None
        board_init[bi] = best_B
        for s in serials:
            if s not in bundles[bi]["obs"]:
                continue
            T_c_b = T_c_ref[s] @ best_B
            rv, _ = cv2.Rodrigues(T_c_b[:3, :3])
            proj, _ = cv2.projectPoints(obj, rv, T_c_b[:3, 3], K[s], dist[s])
            proj = proj.reshape(-1, 2)
            p_raw = bundles[bi]["obs"][s]
            e_norm = float(np.sqrt(np.mean(np.sum((proj - p_raw) ** 2, axis=1))))
            e_flip = float(np.sqrt(np.mean(np.sum((proj - p_raw[::-1]) ** 2, axis=1))))
            if e_flip < e_norm:
                p_raw = p_raw[::-1].copy()
                bundles[bi]["obs"][s] = p_raw
                e_norm = e_flip
            if bundles[bi]["phase"] == "floor" or len(bundles[bi]["obs"]) == 1 or e_norm < 3.5:
                filtered_obs.append((bi, s, p_raw))

    used = sorted({bi for bi, _, _ in filtered_obs})
    board_init = {bi: board_init[bi] for bi in used}
    cam_idx = {s: i for i, s in enumerate(serials[1:])}
    bnd_idx = {bi: i for i, bi in enumerate(used)}
    n_cam = 6 * len(cam_idx)
    x0 = np.zeros(n_cam + 6 * len(used))
    for s, i in cam_idx.items():
        x0[6 * i:6 * i + 3] = Rotation.from_matrix(T_c_ref[s][:3, :3]).as_rotvec()
        x0[6 * i + 3:6 * i + 6] = T_c_ref[s][:3, 3]
    for bi, i in bnd_idx.items():
        x0[n_cam + 6 * i:n_cam + 6 * i + 3] = Rotation.from_matrix(board_init[bi][:3, :3]).as_rotvec()
        x0[n_cam + 6 * i + 3:n_cam + 6 * i + 6] = board_init[bi][:3, 3]
    obs = filtered_obs
    n_pts = len(obj)
    meas = np.concatenate([pts for _, _, pts in obs])
    obs_cam = np.repeat([cam_idx.get(s, -1) for _, s, _ in obs], n_pts)
    obs_bnd = np.repeat([bnd_idx[bi] for bi, _, _ in obs], n_pts)
    obs_serial = np.repeat([serials.index(s) for _, s, _ in obs], n_pts)
    Ks = np.stack([K[s] for s in serials])[obs_serial]
    Ds = np.stack([dist[s] for s in serials])[obs_serial]
    X_board = np.tile(obj, (len(obs), 1))

    def project(x):
        cams = x[:n_cam].reshape(-1, 6)
        bnds = x[n_cam:].reshape(-1, 6)
        Xr = Rotation.from_rotvec(bnds[obs_bnd, :3]).apply(X_board) + bnds[obs_bnd, 3:]
        is_ref = obs_cam < 0
        rv = np.where(is_ref[:, None], 0.0, cams[np.maximum(obs_cam, 0), :3])
        tv = np.where(is_ref[:, None], 0.0, cams[np.maximum(obs_cam, 0), 3:])
        Xc = Rotation.from_rotvec(rv).apply(Xr) + tv
        xn, yn = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
        r2 = xn * xn + yn * yn
        k1, k2, p1, p2, k3 = Ds.T
        radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
        xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
        yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
        u = Ks[:, 0, 0] * xd + Ks[:, 0, 1] * yd + Ks[:, 0, 2]
        v = Ks[:, 1, 1] * yd + Ks[:, 1, 2]
        return np.c_[u, v]

    def residual(x):
        return (project(x) - meas).ravel()

    sparsity = lil_matrix((2 * len(meas), len(x0)), dtype=np.uint8)
    rows = np.arange(len(meas))
    for k in range(6):
        for r in (2 * rows, 2 * rows + 1):
            has = obs_cam >= 0
            sparsity[r[has], 6 * obs_cam[has] + k] = 1
            sparsity[r, n_cam + 6 * obs_bnd + k] = 1
    init_rmse = float(np.sqrt(np.mean(residual(x0).reshape(-1, 2) ** 2) * 2))
    sol = least_squares(residual, x0, jac_sparsity=sparsity, loss="huber", f_scale=1.0, x_scale="jac",
                        method="trf", max_nfev=200)
    err = np.linalg.norm(sol.fun.reshape(-1, 2), axis=1)
    report["bundle_adjustment"] = dict(observations=len(obs), initial_rmse_px=init_rmse,
                                       final_rmse_px=float(np.sqrt(np.mean(err ** 2))), status=int(sol.status))
    for s in serials:
        e = err[obs_serial == serials.index(s)]
        report["cameras"][s].update(bundle_rmse_px=float(np.sqrt(np.mean(e ** 2))), bundle_p99_px=float(np.percentile(e, 99)))
        if s in cam_idx:
            i = cam_idx[s]
            T_c_ref[s] = to_T(sol.x[6 * i:6 * i + 3], sol.x[6 * i + 3:6 * i + 6])
    board_ref = {bi: to_T(sol.x[n_cam + 6 * i:n_cam + 6 * i + 3], sol.x[n_cam + 6 * i + 3:n_cam + 6 * i + 6])
                 for bi, i in bnd_idx.items()}

    floor = [bi for bi in used if bundles[bi]["phase"] == "floor"]
    if not floor:
        raise ValueError("no floor bundle: capture the board lying flat on the floor (--phase floor)")
    floor_pts = np.concatenate([(board_ref[bi][:3, :3] @ obj.T).T + board_ref[bi][:3, 3] for bi in floor])
    centre = floor_pts.mean(axis=0)
    normal = np.linalg.svd(floor_pts - centre)[2][-1]
    cam_centres = np.array([-T[:3, :3].T @ T[:3, 3] for T in (T_c_ref[s] for s in serials)])
    if np.median((cam_centres - centre) @ normal) < 0:
        normal = -normal
    first = board_ref[floor[0]]
    x_axis = first[:3, 0] - (first[:3, 0] @ normal) * normal
    x_axis /= np.linalg.norm(x_axis)
    R_align = np.vstack([x_axis, np.cross(normal, x_axis), normal])
    origin = first[:3, 3] - board_thickness_m * normal
    t_align = -R_align @ origin

    out_cams = {}
    for s in serials:
        R = T_c_ref[s][:3, :3] @ R_align.T
        t = T_c_ref[s][:3, 3] - R @ t_align
        out_cams[s] = dict(serial=s, image_size_wh=list(image_size), K=K[s].tolist(), dist_coeffs=dist[s].tolist(),
                           R_w2c=R.tolist(), t_w2c=t.tolist(), P=(K[s] @ np.c_[R, t]).tolist(),
                           camera_centre_w=(-R.T @ t).tolist(),
                           intrinsic_rmse_px=report["cameras"][s]["intrinsic_rmse_px"],
                           reprojection_rmse_px=report["cameras"][s]["bundle_rmse_px"])

    z_err, edge_err = [], []
    for bi in floor:
        views = [s for s in serials if s in bundles[bi]["obs"]]
        if len(views) < 2:
            continue
        rays = []
        for s in views:
            c = out_cams[s]
            und = cv2.undistortPoints(bundles[bi]["obs"][s].reshape(-1, 1, 2), K[s], dist[s]).reshape(-1, 2)
            rays.append((np.c_[np.array(c["R_w2c"]), np.array(c["t_w2c"])], und))
        X = []
        for j in range(n_pts):
            A = np.concatenate([[u * P[2] - P[0], v * P[2] - P[1]] for P, uv in rays for u, v in [uv[j]]])
            h = np.linalg.svd(A)[2][-1]
            X.append(h[:3] / h[3])
        X = np.array(X)
        z_err.extend(np.abs(X[:, 2] - board_thickness_m) * 1e3)
        grid = X.reshape(pattern[1], pattern[0], 3)
        edges = np.r_[np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel(), np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel()]
        edge_err.extend(np.abs(edges - square_m) * 1e3)
    if not z_err:
        raise ValueError("no floor bundle is seen by two cameras; floor check impossible")
    report["floor_check"] = dict(bundles=len(floor), corners=len(z_err), z_abs_err_mm_max=float(np.max(z_err)),
                                 z_abs_err_mm_median=float(np.median(z_err)),
                                 square_len_err_mm_median=float(np.median(edge_err)),
                                 square_len_err_mm_max=float(np.max(edge_err)))
    calib = dict(schema_version=1, calibration_timestamp_utc=datetime.now(timezone.utc).isoformat(),
                 world="right-handed, Z up, metres; X = floor board column direction; origin = floor board corner 0",
                 pattern=list(pattern), square_m=square_m, board_thickness_m=board_thickness_m,
                 floor_z_max_abs_err_mm=report["floor_check"]["z_abs_err_mm_max"], cameras=out_cams)
    failures = [f"{s} bundle RMSE {r['bundle_rmse_px']:.3f} px" for s, r in report["cameras"].items()
                if r["bundle_rmse_px"] >= max_rmse_px]
    if report["floor_check"]["z_abs_err_mm_max"] >= max_floor_err_mm:
        failures.append(f"floor Z error {report['floor_check']['z_abs_err_mm_max']:.2f} mm")
    report["failures"] = failures
    return calib, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True, help="folder name under data/calib_capture")
    ap.add_argument("--square-mm", type=float, required=True, help="measured square edge length in millimetres")
    ap.add_argument("--board-thickness-mm", type=float, default=0.0)
    ap.add_argument("--free-k3", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "calibration" / "physical_cameras.json")
    args = ap.parse_args()
    bundles = load_session(ROOT / "data" / "calib_capture" / args.session)
    serials = bundles[0]["serials"]
    calib, report = calibrate(bundles, serials, bundles[0]["size"], bundles[0]["pattern"], args.square_mm * 1e-3,
                              board_thickness_m=args.board_thickness_mm * 1e-3, free_k3=args.free_k3)
    report["detections"] = {b["name"]: sorted(b["obs"]) for b in bundles}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    (args.out.parent / "calibration_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("cameras", "bundle_adjustment", "floor_check", "failures")}, indent=1))
    if report["failures"]:
        raise SystemExit("calibration rejected: " + "; ".join(report["failures"]))
    args.out.write_text(json.dumps(calib, indent=2), encoding="utf-8")
    print(f"written {args.out}")
    if list(bundles[0]["size"]) == [1624, 1240]:
        bin2 = json.loads(json.dumps(calib))
        for cam in bin2["cameras"].values():
            cam["image_size_wh"] = [812, 620]
            K_b2 = np.array(cam["K"], dtype=np.float64)
            K_b2[:2, :] *= 0.5
            P_b2 = K_b2 @ np.column_stack([np.array(cam["R_w2c"]), np.array(cam["t_w2c"])])
            cam["K"] = K_b2.tolist()
            cam["P"] = P_b2.tolist()
        bin2_out = args.out.with_name("physical_cameras_bin2_812x620.json")
        bin2_out.write_text(json.dumps(bin2, indent=2), encoding="utf-8")
        print(f"written {bin2_out}")


if __name__ == "__main__":
    main()
