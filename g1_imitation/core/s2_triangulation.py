"""Module S2: Multi-View Confidence-Squared Weighted DLT Triangulation & Outlier Rejection.

Specification Reference: `docs/方案规范.md` — 模块 S2
Input:
  - `keypoints_2d`: np.ndarray of shape (N_cams, 27, 3) with [u, v, c] (pixels & confidence in [0, 1]).
  - `proj_matrices`: np.ndarray of shape (N_cams, 3, 4), camera projection matrices P_k = K_k [R_k | t_k].
Output:
  - `TriangulationResult`:
    - `points_3d_m`: np.ndarray of shape (27, 3) in right-handed Z-up world frame (units: metres).
      Unobservable points (< 2 valid cameras) are filled with `np.nan`.
    - `valid_mask`: np.ndarray of shape (27,) bool.
    - `inlier_camera_counts`: np.ndarray of shape (27,) int32.
    - `reprojection_errors_px`: np.ndarray of shape (27,) float64 (mean inlier reprojection error in pixels).
    - `solve_time_ms`: float (wall-clock execution time in milliseconds).
Failure Behavior:
  - Any joint with fewer than 2 valid cameras (c >= c_min) outputs `np.nan` for that joint without crashing.
  - Inter-frame displacement exceeding `max_displacement_m = 0.25 m` (at 100 Hz) is directionally clamped.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

from numba import njit
import numpy as np


@njit(cache=True, fastmath=True)
def _solve_weighted_dlt_fast3x3(
    uvc_j: np.ndarray,
    proj_matrices: np.ndarray,
    subset_indices: np.ndarray,
    subset_count: int,
) -> tuple[float, float, float, bool]:
    """Closed-form 3x3 normal equation solver for rapid subset consensus search (zero heap allocation)."""
    if subset_count < 2:
        return np.nan, np.nan, np.nan, False

    m00 = 0.0; m01 = 0.0; m02 = 0.0
    m11 = 0.0; m12 = 0.0; m22 = 0.0
    r0 = 0.0; r1 = 0.0; r2 = 0.0

    for idx in range(subset_count):
        k = subset_indices[idx]
        u = uvc_j[k, 0]
        v = uvc_j[k, 1]
        c = uvc_j[k, 2]
        w2 = (c * c) * (c * c)
        P = proj_matrices[k]

        ax0 = u * P[2, 0] - P[0, 0]
        ax1 = u * P[2, 1] - P[0, 1]
        ax2 = u * P[2, 2] - P[0, 2]
        bx = P[0, 3] - u * P[2, 3]

        ay0 = v * P[2, 0] - P[1, 0]
        ay1 = v * P[2, 1] - P[1, 1]
        ay2 = v * P[2, 2] - P[1, 2]
        by = P[1, 3] - v * P[2, 3]

        m00 += w2 * (ax0 * ax0 + ay0 * ay0)
        m01 += w2 * (ax0 * ax1 + ay0 * ay1)
        m02 += w2 * (ax0 * ax2 + ay0 * ay2)
        m11 += w2 * (ax1 * ax1 + ay1 * ay1)
        m12 += w2 * (ax1 * ax2 + ay1 * ay2)
        m22 += w2 * (ax2 * ax2 + ay2 * ay2)

        r0 += w2 * (ax0 * bx + ay0 * by)
        r1 += w2 * (ax1 * bx + ay1 * by)
        r2 += w2 * (ax2 * bx + ay2 * by)

    c00 = m11 * m22 - m12 * m12
    c01 = -(m01 * m22 - m12 * m02)
    c02 = m01 * m12 - m11 * m02
    det = m00 * c00 + m01 * c01 + m02 * c02
    if abs(det) < 1e-12:
        return np.nan, np.nan, np.nan, False

    inv_det = 1.0 / det
    c11 = m00 * m22 - m02 * m02
    c12 = -(m00 * m12 - m01 * m02)
    c22 = m00 * m11 - m01 * m01

    x = (c00 * r0 + c01 * r1 + c02 * r2) * inv_det
    y = (c01 * r0 + c11 * r1 + c12 * r2) * inv_det
    z = (c02 * r0 + c12 * r1 + c22 * r2) * inv_det
    return x, y, z, True


@njit(cache=True, fastmath=True)
def _solve_weighted_dlt_subset(
    uvc_j: np.ndarray,
    proj_matrices: np.ndarray,
    subset_indices: np.ndarray,
    subset_count: int,
) -> tuple[float, float, float, bool]:
    """Solve confidence-squared weighted DLT (S2.2.1) for a single joint on `subset_indices[:subset_count]`."""
    if subset_count < 2:
        return np.nan, np.nan, np.nan, False

    A = np.zeros((2 * subset_count, 4), dtype=np.float64)
    for idx in range(subset_count):
        k = subset_indices[idx]
        u = uvc_j[k, 0]
        v = uvc_j[k, 1]
        c = uvc_j[k, 2]
        w = c * c
        P = proj_matrices[k]
        for col in range(4):
            A[2 * idx, col] = w * (u * P[2, col] - P[0, col])
            A[2 * idx + 1, col] = w * (v * P[2, col] - P[1, col])

    _, _, vt = np.linalg.svd(A, full_matrices=False)
    X_h = vt[3]
    if abs(X_h[3]) < 1e-10:
        return np.nan, np.nan, np.nan, False
    inv_w = 1.0 / X_h[3]
    return X_h[0] * inv_w, X_h[1] * inv_w, X_h[2] * inv_w, True


@njit(cache=True, fastmath=True)
def _reproj_error_cam(
    x: float,
    y: float,
    z: float,
    u: float,
    v: float,
    P: np.ndarray,
) -> float:
    """Compute Euclidean pixel reprojection error e_k for camera P."""
    zc = P[2, 0] * x + P[2, 1] * y + P[2, 2] * z + P[2, 3]
    if zc <= 1e-6:
        return 1e9
    inv_zc = 1.0 / zc
    u_hat = (P[0, 0] * x + P[0, 1] * y + P[0, 2] * z + P[0, 3]) * inv_zc
    v_hat = (P[1, 0] * x + P[1, 1] * y + P[1, 2] * z + P[1, 3]) * inv_zc
    du = u - u_hat
    dv = v - v_hat
    return np.sqrt(du * du + dv * dv)


@njit(cache=True, fastmath=True)
def triangulate_frame_numba(
    keypoints_2d: np.ndarray,
    proj_matrices: np.ndarray,
    prev_points_3d: np.ndarray,
    has_prev: bool,
    c_min: float = 0.25,
    tau_reproj_px: float = 15.0,
    max_displacement_m: float = 0.25,
    use_consensus_search: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Numba-accelerated 27-point multi-view weighted DLT + continuous Cauchy IRLS + 3x3 covariance."""
    n_cams = keypoints_2d.shape[0]
    n_joints = keypoints_2d.shape[1]

    out_pts = np.full((n_joints, 3), np.nan, dtype=np.float64)
    inlier_counts = np.zeros(n_joints, dtype=np.int32)
    mean_reproj_px = np.full(n_joints, np.nan, dtype=np.float64)
    out_covs = np.full((n_joints, 3, 3), np.nan, dtype=np.float64)

    valid_cams = np.empty(n_cams, dtype=np.int32)
    inlier_cams = np.empty(n_cams, dtype=np.int32)
    trial_cams = np.empty(n_cams, dtype=np.int32)
    cam_errors = np.empty(n_cams, dtype=np.float64)

    max_cands = 1 + n_cams + (n_cams * (n_cams - 1)) // 2
    cand_x = np.empty(max_cands, dtype=np.float64)
    cand_y = np.empty(max_cands, dtype=np.float64)
    cand_z = np.empty(max_cands, dtype=np.float64)
    cand_loss = np.empty(max_cands, dtype=np.float64)

    tau_sq = tau_reproj_px * tau_reproj_px
    sigma0_px_sq = 4.0  # nominal 2.0 px detector baseline variance

    for j in range(n_joints):
        uvc_j = keypoints_2d[:, j, :]
        m_valid = 0
        for k in range(n_cams):
            u = uvc_j[k, 0]
            v = uvc_j[k, 1]
            c = uvc_j[k, 2]
            if c >= c_min and np.isfinite(u) and np.isfinite(v):
                valid_cams[m_valid] = k
                m_valid += 1

        if m_valid < 2:
            continue

        if not use_consensus_search:
            # Literal Section S2.2.2 two-round residual thresholding around Round 1 SVD DLT solution
            x0, y0, z0, ok0 = _solve_weighted_dlt_subset(uvc_j, proj_matrices, valid_cams, m_valid)
            if not ok0:
                continue

            max_err = 0.0
            for idx in range(m_valid):
                k = valid_cams[idx]
                err = _reproj_error_cam(x0, y0, z0, uvc_j[k, 0], uvc_j[k, 1], proj_matrices[k])
                cam_errors[idx] = err
                if err > max_err:
                    max_err = err

            active_count = m_valid
            for idx in range(m_valid):
                inlier_cams[idx] = valid_cams[idx]

            if max_err > tau_reproj_px and m_valid >= 3:
                m_inliers = 0
                for idx in range(m_valid):
                    if cam_errors[idx] <= tau_reproj_px:
                        inlier_cams[m_inliers] = valid_cams[idx]
                        m_inliers += 1
                if m_inliers >= 2:
                    xf, yf, zf, okf = _solve_weighted_dlt_subset(uvc_j, proj_matrices, inlier_cams, m_inliers)
                    if okf:
                        x0, y0, z0 = xf, yf, zf
                        active_count = m_inliers

            inlier_counts[j] = active_count
            err_sum = 0.0
            m00 = 0.0; m01 = 0.0; m02 = 0.0
            m11 = 0.0; m12 = 0.0; m22 = 0.0
            for idx in range(active_count):
                k = inlier_cams[idx]
                u = uvc_j[k, 0]
                v = uvc_j[k, 1]
                c = uvc_j[k, 2]
                P = proj_matrices[k]
                err_sum += _reproj_error_cam(x0, y0, z0, u, v, P)
                zc = max(P[2, 0] * x0 + P[2, 1] * y0 + P[2, 2] * z0 + P[2, 3], 0.1)
                inv_zc = 1.0 / zc
                w = (c * c) * (c * c) * (inv_zc * inv_zc)
                ax0 = u * P[2, 0] - P[0, 0]
                ax1 = u * P[2, 1] - P[0, 1]
                ax2 = u * P[2, 2] - P[0, 2]
                ay0 = v * P[2, 0] - P[1, 0]
                ay1 = v * P[2, 1] - P[1, 1]
                ay2 = v * P[2, 2] - P[1, 2]
                m00 += w * (ax0 * ax0 + ay0 * ay0)
                m01 += w * (ax0 * ax1 + ay0 * ay1)
                m02 += w * (ax0 * ax2 + ay0 * ay2)
                m11 += w * (ax1 * ax1 + ay1 * ay1)
                m12 += w * (ax1 * ax2 + ay1 * ay2)
                m22 += w * (ax2 * ax2 + ay2 * ay2)
            mean_e = err_sum / active_count
            mean_reproj_px[j] = mean_e
            c00 = m11 * m22 - m12 * m12
            c01 = -(m01 * m22 - m12 * m02)
            c02 = m01 * m12 - m11 * m02
            det = m00 * c00 + m01 * c01 + m02 * c02
            if abs(det) > 1e-12:
                scale_cov = (sigma0_px_sq + mean_e * mean_e) / det
                c11 = m00 * m22 - m02 * m02
                c12 = -(m00 * m12 - m01 * m02)
                c22 = m00 * m11 - m01 * m01
                out_covs[j, 0, 0] = c00 * scale_cov
                out_covs[j, 0, 1] = c01 * scale_cov
                out_covs[j, 0, 2] = c02 * scale_cov
                out_covs[j, 1, 0] = c01 * scale_cov
                out_covs[j, 1, 1] = c11 * scale_cov
                out_covs[j, 1, 2] = c12 * scale_cov
                out_covs[j, 2, 0] = c02 * scale_cov
                out_covs[j, 2, 1] = c12 * scale_cov
                out_covs[j, 2, 2] = c22 * scale_cov
        else:
            # Continuous Residual-Weighted Robust IRLS with Smooth Softmin Consensus Seed
            n_cands = 0
            best_loss = 1e18
            best_second_err = 1e18

            x_all, y_all, z_all, ok_all = _solve_weighted_dlt_fast3x3(uvc_j, proj_matrices, valid_cams, m_valid)
            if ok_all:
                loss = 0.0
                e_max1 = 0.0
                e_max2 = 0.0
                for idx in range(m_valid):
                    k = valid_cams[idx]
                    e = _reproj_error_cam(x_all, y_all, z_all, uvc_j[k, 0], uvc_j[k, 1], proj_matrices[k])
                    if e > e_max1:
                        e_max2 = e_max1
                        e_max1 = e
                    elif e > e_max2:
                        e_max2 = e
                    r2 = (e * e) / tau_sq
                    c = uvc_j[k, 2]
                    loss += (c * c) * (r2 / (1.0 + r2))
                cand_x[n_cands] = x_all
                cand_y[n_cands] = y_all
                cand_z[n_cands] = z_all
                cand_loss[n_cands] = loss
                n_cands += 1
                best_loss = loss
                best_second_err = e_max2

            if m_valid >= 3:
                for drop1 in range(m_valid):
                    t_cnt = 0
                    for idx in range(m_valid):
                        if idx != drop1:
                            trial_cams[t_cnt] = valid_cams[idx]
                            t_cnt += 1
                    xt, yt, zt, okt = _solve_weighted_dlt_fast3x3(uvc_j, proj_matrices, trial_cams, t_cnt)
                    if not okt:
                        continue
                    loss = 0.0
                    e_max1 = 0.0
                    e_max2 = 0.0
                    for idx in range(m_valid):
                        k = valid_cams[idx]
                        e = _reproj_error_cam(xt, yt, zt, uvc_j[k, 0], uvc_j[k, 1], proj_matrices[k])
                        if e > e_max1:
                            e_max2 = e_max1
                            e_max1 = e
                        elif e > e_max2:
                            e_max2 = e
                        r2 = (e * e) / tau_sq
                        c = uvc_j[k, 2]
                        loss += (c * c) * (r2 / (1.0 + r2))
                    cand_x[n_cands] = xt
                    cand_y[n_cands] = yt
                    cand_z[n_cands] = zt
                    cand_loss[n_cands] = loss
                    n_cands += 1
                    if loss < best_loss:
                        best_loss = loss
                        best_second_err = e_max2

            if m_valid >= 4 and best_second_err > 0.45 * tau_reproj_px:
                for drop1 in range(m_valid - 1):
                    for drop2 in range(drop1 + 1, m_valid):
                        t_cnt = 0
                        for idx in range(m_valid):
                            if idx != drop1 and idx != drop2:
                                trial_cams[t_cnt] = valid_cams[idx]
                                t_cnt += 1
                        xt, yt, zt, okt = _solve_weighted_dlt_fast3x3(uvc_j, proj_matrices, trial_cams, t_cnt)
                        if not okt:
                            continue
                        loss = 0.0
                        for idx in range(m_valid):
                            k = valid_cams[idx]
                            e = _reproj_error_cam(xt, yt, zt, uvc_j[k, 0], uvc_j[k, 1], proj_matrices[k])
                            r2 = (e * e) / tau_sq
                            c = uvc_j[k, 2]
                            loss += (c * c) * (r2 / (1.0 + r2))
                        cand_x[n_cands] = xt
                        cand_y[n_cands] = yt
                        cand_z[n_cands] = zt
                        cand_loss[n_cands] = loss
                        n_cands += 1
                        if loss < best_loss:
                            best_loss = loss

            if n_cands == 0:
                continue

            # Smooth softmin combination across candidate seeds
            sum_w = 0.0
            x0 = 0.0
            y0 = 0.0
            z0 = 0.0
            for i_c in range(n_cands):
                dl = cand_loss[i_c] - best_loss
                if dl < 2.5:
                    w_s = np.exp(-15.0 * dl)
                    x0 += w_s * cand_x[i_c]
                    y0 += w_s * cand_y[i_c]
                    z0 += w_s * cand_z[i_c]
                    sum_w += w_s
            if sum_w <= 1e-12:
                continue
            x0 /= sum_w
            y0 /= sum_w
            z0 /= sum_w

            # 3 iterations of continuous Cauchy-rational IRLS over all valid cameras
            c00 = 0.0; c01 = 0.0; c02 = 0.0
            c11 = 0.0; c12 = 0.0; c22 = 0.0
            det = 0.0
            for _ in range(3):
                m00 = 0.0; m01 = 0.0; m02 = 0.0
                m11 = 0.0; m12 = 0.0; m22 = 0.0
                r0 = 0.0; r1 = 0.0; r2 = 0.0
                for idx in range(m_valid):
                    k = valid_cams[idx]
                    u = uvc_j[k, 0]
                    v = uvc_j[k, 1]
                    c = uvc_j[k, 2]
                    P = proj_matrices[k]
                    zc = max(P[2, 0] * x0 + P[2, 1] * y0 + P[2, 2] * z0 + P[2, 3], 0.1)
                    inv_zc = 1.0 / zc
                    u_hat = (P[0, 0] * x0 + P[0, 1] * y0 + P[0, 2] * z0 + P[0, 3]) * inv_zc
                    v_hat = (P[1, 0] * x0 + P[1, 1] * y0 + P[1, 2] * z0 + P[1, 3]) * inv_zc
                    du = u - u_hat
                    dv = v - v_hat
                    r2_rat = (du * du + dv * dv) / tau_sq
                    denom = 1.0 + r2_rat
                    w_rob = 1.0 / (denom * denom * denom)
                    w = (c * c) * (c * c) * w_rob * (inv_zc * inv_zc)

                    ax0 = u * P[2, 0] - P[0, 0]
                    ax1 = u * P[2, 1] - P[0, 1]
                    ax2 = u * P[2, 2] - P[0, 2]
                    bx_k = P[0, 3] - u * P[2, 3]

                    ay0 = v * P[2, 0] - P[1, 0]
                    ay1 = v * P[2, 1] - P[1, 1]
                    ay2 = v * P[2, 2] - P[1, 2]
                    by_k = P[1, 3] - v * P[2, 3]

                    m00 += w * (ax0 * ax0 + ay0 * ay0)
                    m01 += w * (ax0 * ax1 + ay0 * ay1)
                    m02 += w * (ax0 * ax2 + ay0 * ay2)
                    m11 += w * (ax1 * ax1 + ay1 * ay1)
                    m12 += w * (ax1 * ax2 + ay1 * ay2)
                    m22 += w * (ax2 * ax2 + ay2 * ay2)

                    r0 += w * (ax0 * bx_k + ay0 * by_k)
                    r1 += w * (ax1 * bx_k + ay1 * by_k)
                    r2 += w * (ax2 * bx_k + ay2 * by_k)

                c00 = m11 * m22 - m12 * m12
                c01 = -(m01 * m22 - m12 * m02)
                c02 = m01 * m12 - m11 * m02
                det = m00 * c00 + m01 * c01 + m02 * c02
                if abs(det) < 1e-12:
                    break
                inv_det = 1.0 / det
                c11 = m00 * m22 - m02 * m02
                c12 = -(m00 * m12 - m01 * m02)
                c22 = m00 * m11 - m01 * m01

                x0 = (c00 * r0 + c01 * r1 + c02 * r2) * inv_det
                y0 = (c01 * r0 + c11 * r1 + c12 * r2) * inv_det
                z0 = (c02 * r0 + c12 * r1 + c22 * r2) * inv_det

            # Compute effective inlier count, weighted reprojection residual, and 3x3 covariance
            w_err_sum = 0.0
            w_sum = 0.0
            eff_inliers = 0
            for idx in range(m_valid):
                k = valid_cams[idx]
                c = uvc_j[k, 2]
                e = _reproj_error_cam(x0, y0, z0, uvc_j[k, 0], uvc_j[k, 1], proj_matrices[k])
                if e <= tau_reproj_px:
                    eff_inliers += 1
                r2_rat = (e * e) / tau_sq
                denom = 1.0 + r2_rat
                w_px = (c * c) * (c * c) / (denom * denom * denom)
                w_err_sum += w_px * e
                w_sum += w_px

            inlier_counts[j] = max(2, eff_inliers)
            mean_e = (w_err_sum / w_sum) if w_sum > 1e-12 else 0.0
            mean_reproj_px[j] = mean_e

            if abs(det) >= 1e-12:
                scale_cov = (sigma0_px_sq + mean_e * mean_e) / det
                out_covs[j, 0, 0] = c00 * scale_cov
                out_covs[j, 0, 1] = c01 * scale_cov
                out_covs[j, 0, 2] = c02 * scale_cov
                out_covs[j, 1, 0] = c01 * scale_cov
                out_covs[j, 1, 1] = c11 * scale_cov
                out_covs[j, 1, 2] = c12 * scale_cov
                out_covs[j, 2, 0] = c02 * scale_cov
                out_covs[j, 2, 1] = c12 * scale_cov
                out_covs[j, 2, 2] = c22 * scale_cov

        # Step S2.2.3: Inter-frame max displacement clamp (d_max = 0.25 m)
        if has_prev:
            px = prev_points_3d[j, 0]
            py = prev_points_3d[j, 1]
            pz = prev_points_3d[j, 2]
            if np.isfinite(px) and np.isfinite(py) and np.isfinite(pz):
                dx = x0 - px
                dy = y0 - py
                dz = z0 - pz
                dist = np.sqrt(dx * dx + dy * dy + dz * dz)
                if dist > max_displacement_m:
                    scale = max_displacement_m / dist
                    x0 = px + dx * scale
                    y0 = py + dy * scale
                    z0 = pz + dz * scale

        out_pts[j, 0] = x0
        out_pts[j, 1] = y0
        out_pts[j, 2] = z0

    return out_pts, inlier_counts, mean_reproj_px, out_covs


@dataclass(frozen=True)
class TriangulationResult:
    """Output of Module S2 for a single frame."""
    points_3d_m: np.ndarray
    valid_mask: np.ndarray
    inlier_camera_counts: np.ndarray
    reprojection_errors_px: np.ndarray
    solve_time_ms: float
    point_covariances_m2: np.ndarray | None = None


class MultiViewTriangulator:
    """Stateful wrapper for Module S2: Numba weighted DLT + outlier rejection + velocity clamp."""

    def __init__(
        self,
        c_min: float = 0.25,
        reproj_outlier_threshold_px: float = 15.0,
        max_displacement_m: float = 0.25,
        use_consensus_search: bool = True,
    ) -> None:
        self.c_min = float(c_min)
        self.reproj_outlier_threshold_px = float(reproj_outlier_threshold_px)
        self.max_displacement_m = float(max_displacement_m)
        self.use_consensus_search = bool(use_consensus_search)
        self.prev_points_3d: np.ndarray = np.full((27, 3), np.nan, dtype=np.float64)
        self.has_prev: bool = False
        self._warmup()

    def _warmup(self) -> None:
        dummy_2d = np.ones((6, 27, 3), dtype=np.float64)
        dummy_P = np.zeros((6, 3, 4), dtype=np.float64)
        for k in range(6):
            dummy_P[k, 0, 0] = 1000.0
            dummy_P[k, 1, 1] = 1000.0
            dummy_P[k, 2, 2] = 1.0
            dummy_P[k, 2, 3] = 3.0 + 0.1 * k
        for consensus_flag in (False, True):
            _ = triangulate_frame_numba(
                dummy_2d,
                dummy_P,
                self.prev_points_3d,
                False,
                self.c_min,
                self.reproj_outlier_threshold_px,
                self.max_displacement_m,
                consensus_flag,
            )

    def reset(self) -> None:
        self.prev_points_3d = np.full((27, 3), np.nan, dtype=np.float64)
        self.has_prev = False

    def process_frame(
        self,
        keypoints_2d: np.ndarray,
        proj_matrices: np.ndarray,
        update_history: bool = True,
    ) -> TriangulationResult:
        kp2d = np.ascontiguousarray(keypoints_2d, dtype=np.float64)
        P_mats = np.ascontiguousarray(proj_matrices, dtype=np.float64)
        if kp2d.ndim != 3 or kp2d.shape[1:] != (27, 3):
            raise ValueError(f"Expected keypoints_2d of shape (N_cams, 27, 3), got {kp2d.shape}")
        if P_mats.shape != (kp2d.shape[0], 3, 4):
            raise ValueError(f"Expected proj_matrices of shape ({kp2d.shape[0]}, 3, 4), got {P_mats.shape}")

        t0 = time.perf_counter()
        pts_3d, inlier_counts, reproj_px, covs_m2 = triangulate_frame_numba(
            kp2d,
            P_mats,
            self.prev_points_3d,
            self.has_prev,
            self.c_min,
            self.reproj_outlier_threshold_px,
            self.max_displacement_m,
            self.use_consensus_search,
        )
        solve_ms = (time.perf_counter() - t0) * 1000.0

        valid_mask = np.all(np.isfinite(pts_3d), axis=1)
        if update_history:
            for j in range(27):
                if valid_mask[j]:
                    self.prev_points_3d[j] = pts_3d[j]
            if np.any(valid_mask):
                self.has_prev = True

        return TriangulationResult(
            points_3d_m=pts_3d,
            valid_mask=valid_mask,
            inlier_camera_counts=inlier_counts,
            reprojection_errors_px=reproj_px,
            solve_time_ms=float(solve_ms),
            point_covariances_m2=covs_m2,
        )


def create_circular_camera_rig(
    num_cameras: int = 6,
    radius_m: float = 3.2,
    camera_height_m: float = 1.35,
    lookat_xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.90),
    image_width: int = 1280,
    image_height: int = 720,
    focal_px: float = 950.0,
) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
    """Create a calibrated circular multi-camera array around `lookat_xyz_m`.

    Returns:
      - `proj_matrices`: (N_cams, 3, 4) projection matrices P_k = K_k [R_k | t_k].
      - `camera_params`: list of dicts containing K, R_w2c, t_w2c, cam_pos_w, P.
    """
    K = np.array(
        [
            [focal_px, 0.0, image_width * 0.5],
            [0.0, focal_px, image_height * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    lookat = np.asarray(lookat_xyz_m, dtype=np.float64)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    proj_matrices = np.zeros((num_cameras, 3, 4), dtype=np.float64)
    cam_params: list[dict[str, np.ndarray]] = []
    for k in range(num_cameras):
        theta = 2.0 * np.pi * k / num_cameras
        cam_pos = np.array(
            [lookat[0] + radius_m * np.cos(theta), lookat[1] + radius_m * np.sin(theta), camera_height_m],
            dtype=np.float64,
        )
        # OpenCV camera frame: +Z forward (towards lookat), +X right, +Y down
        z_cam = lookat - cam_pos
        z_cam /= float(np.linalg.norm(z_cam))
        x_cam = np.cross(z_cam, world_up)
        x_cam /= float(np.linalg.norm(x_cam))
        y_cam = np.cross(z_cam, x_cam)
        y_cam /= float(np.linalg.norm(y_cam))

        R_w2c = np.vstack((x_cam, y_cam, z_cam))
        t_w2c = -R_w2c @ cam_pos
        Rt = np.column_stack((R_w2c, t_w2c))
        P = K @ Rt
        proj_matrices[k] = P
        cam_params.append(
            {
                "K": K.copy(),
                "R_w2c": R_w2c,
                "t_w2c": t_w2c,
                "cam_pos_w": cam_pos,
                "P": P,
            }
        )
    return proj_matrices, cam_params


def project_points_3d_to_2d(
    points_3d_m: np.ndarray,
    proj_matrices: np.ndarray,
    default_confidence: float = 0.92,
) -> np.ndarray:
    """Project (N_pts, 3) 3D world points into (N_cams, N_pts, 3) [u, v, c] pixel coordinates."""
    pts = np.asarray(points_3d_m, dtype=np.float64)
    P_mats = np.asarray(proj_matrices, dtype=np.float64)
    n_cams = P_mats.shape[0]
    n_pts = pts.shape[0]
    out = np.zeros((n_cams, n_pts, 3), dtype=np.float64)
    pts_h = np.column_stack((pts, np.ones(n_pts, dtype=np.float64)))
    for k in range(n_cams):
        uvw = (P_mats[k] @ pts_h.T).T
        z = np.maximum(uvw[:, 2], 1e-6)
        out[k, :, 0] = uvw[:, 0] / z
        out[k, :, 1] = uvw[:, 1] / z
        out[k, :, 2] = default_confidence
    return out
