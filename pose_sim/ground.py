"""Static scene ground, independent of human landmarks and robot models.

Fit input: finite multiview scene points in calibrated Z-up metres. Output:
unit normal n, offset d (n @ X + d = 0), observed XY domain, residual bound,
and calibration identity. A robust plane is a scene hypothesis, not force truth.
Bad tilt, insufficient spatial extent/inliers, or poor held-out agreement fail.
"""
import hashlib
import json
import numpy as np


def calibration_hash(calibration):
    return hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()


def fit_ground(points, calibration_sha256, *, seed=19, tolerance_m=.025,
               minimum_points=8, minimum_span_m=.4, maximum_tilt_deg=10.):
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError('Finite scene XYZ required')
    # Repeated features at different rectification seeds are not independent.
    unique = []
    for point in p:
        if not unique or np.min(np.linalg.norm(np.array(unique)-point, axis=1)) > .08:
            unique.append(point)
    p = np.asarray(unique)
    if len(p) < minimum_points + 3:
        raise ValueError('Insufficient independent scene points')
    # Spatial features, not repeated views of the same point, are held out.
    order = np.random.default_rng(seed).permutation(len(p))
    test = p[order[::4]]
    train = p[np.delete(order, np.arange(0, len(order), 4))]
    best = None
    rng = np.random.default_rng(seed)
    for _ in range(400):
        sample = train[rng.choice(len(train), 3, replace=False)]
        normal = np.cross(sample[1]-sample[0], sample[2]-sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal /= norm
        if normal[2] < 0:
            normal *= -1
        if normal[2] < np.cos(np.radians(maximum_tilt_deg)):
            continue
        d = -normal @ sample[0]
        err = np.abs(train @ normal + d)
        ids = err <= tolerance_m
        key = (int(ids.sum()), -float(np.median(err[ids])))
        if best is None or key > best[0]:
            best = key, ids
    if best is None or best[0][0] < minimum_points:
        raise ValueError('No sufficiently supported near-horizontal plane')
    inliers = train[best[1]]
    centre = inliers.mean(axis=0)
    _, _, vt = np.linalg.svd(inliers-centre)
    normal = vt[-1] * (1 if vt[-1, 2] > 0 else -1)
    d = -normal @ centre
    spread = np.linalg.svd(inliers[:, :2]-centre[:2], compute_uv=False)/np.sqrt(len(inliers))
    if spread[-1] < minimum_span_m/2 or normal[2] < np.cos(np.radians(maximum_tilt_deg)):
        raise ValueError('Degenerate coverage or excessive floor tilt')
    test_error = np.abs(test @ normal + d)
    # RANSAC outliers are retained in validation, never silently discarded.
    if np.mean(test_error <= tolerance_m*2) < .75:
        raise ValueError('Scene plane fails held-out feature check')
    residual = np.abs(inliers @ normal + d)
    return dict(schema_version=1, status='scene_plane_hypothesis', units='metre',
        world_axes='calibrated_Z_up', calibration_sha256=calibration_sha256,
        normal_world=normal.tolist(), offset_m=float(d),
        uncertainty_bound_m=float(max(.01, np.percentile(residual, 95),
                                     np.percentile(test_error[test_error <= tolerance_m*2], 95))),
        uncertainty_calibrated=False, physical_support_verified=False,
        xy_min_m=inliers[:, :2].min(axis=0).tolist(), xy_max_m=inliers[:, :2].max(axis=0).tolist(),
        inference_basis='Natural static scene features only; no human pose/contact labels',
        diagnostics=dict(unique_features=len(p), training_features=len(train), inliers=len(inliers),
            heldout_features=len(test), heldout_within_50mm_fraction=float(np.mean(test_error <= .05)),
            heldout_error_m=test_error.tolist(), fit_p95_m=float(np.percentile(residual, 95)),
            xy_spread_m=spread.tolist(), tilt_deg=float(np.degrees(np.arccos(normal[2])))))


def validate_ground(ground, calibration_sha256):
    if not isinstance(ground, dict) or ground.get('calibration_sha256') != calibration_sha256:
        raise ValueError('Missing ground or mismatched calibration')
    n = np.asarray(ground['normal_world'], float)
    d, bound = float(ground['offset_m']), float(ground['uncertainty_bound_m'])
    lo, hi = np.asarray(ground['xy_min_m'], float), np.asarray(ground['xy_max_m'], float)
    if (n.shape != (3,) or lo.shape != (2,) or hi.shape != (2,) or
            not np.isfinite(np.r_[n, d, bound, lo, hi]).all() or
            not np.isclose(np.linalg.norm(n), 1., atol=1e-6) or n[2] < .98 or
            not 0 < bound <= .03 or np.any(hi <= lo) or ground.get('units') != 'metre'):
        raise ValueError('Invalid or uncertain ground geometry')
    return n, d, bound, lo, hi
