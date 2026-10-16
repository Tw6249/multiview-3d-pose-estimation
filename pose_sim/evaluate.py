"""Evaluation only: comparison to exported rig joint proxies and oracle sanity check."""
import argparse
import json
from pathlib import Path
import numpy as np
from .geometry import JOINTS, triangulate
from .pipeline import read_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument('session', type=Path)
    a = p.parse_args()
    s = a.session
    cams = json.loads((s/'cameras.json').read_text())['cameras']
    preds = {r['frame_id']: r for r in read_jsonl(s/'poses3d.jsonl')}
    truth = list(read_jsonl(s/'ground_truth.jsonl'))
    errors, per_joint, oracle_errors = [], {j: [] for j in JOINTS}, []
    total = 0
    for row in truth:
        pred = preds[row['frame_id']]
        for j, gt in enumerate(row['joints_world_m']):
            if gt is None:
                continue
            total += 1
            point = pred['joints_world_m'][j]
            if point is not None:
                error = float(np.linalg.norm(np.array(point)-gt)*1000)
                errors.append(error)
                per_joint[JOINTS[j]].append(error)
            ids = list(cams)
            oracle = triangulate([cams[c] for c in ids],
                [row['projected_joints_px'][c][j] for c in ids], np.ones(len(ids)))
            if oracle:
                oracle_errors.append(float(np.linalg.norm(np.array(oracle['xyz_m'])-gt)*1000))
    valid_all = sum(sum(x is not None for x in r['joints_world_m']) for r in preds.values())
    reproj = [q['mean_reprojection_px'] for r in preds.values() for q in r['quality'] if q]
    result = {
        'frames': len(preds), 'camera_count': len(cams),
        'detected_joint_reconstruction_rate': valid_all/(17*len(preds)),
        'body12_proxy_coverage': len(errors)/total if total else 0,
        'body12_proxy_mpjpe_mm': float(np.mean(errors)) if errors else None,
        'body12_proxy_median_mm': float(np.median(errors)) if errors else None,
        'body12_proxy_p95_mm': float(np.percentile(errors, 95)) if errors else None,
        'mean_inlier_reprojection_px': float(np.mean(reproj)) if reproj else None,
        'oracle_geometry_max_error_mm': max(oracle_errors) if oracle_errors else None,
        'oracle_valid_fraction': len(oracle_errors)/total if total else 0,
        'per_joint_proxy_error_mm': {j: float(np.mean(e)) if e else None for j, e in per_joint.items()},
        'limitations': ['Stylized CesiumMan sample; not a real-world accuracy benchmark.',
            'Only 12 rig joint centres have proxy ground truth; face landmarks have no GT.',
            'COCO anatomical landmarks and rig centres differ; proxy MPJPE includes definition mismatch.',
            'Oracle projection test only verifies geometry and never substitutes for image detections.',
            'One person, fixed calibrated cameras, ideal global-shutter pinhole render; no real-device validation yet.']}
    (s/'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
