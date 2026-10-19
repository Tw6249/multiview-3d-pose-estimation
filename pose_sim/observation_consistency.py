"""Current-image consistency checks, independent of tracked pose and body size.

Input: named pixels/confidences, calibrated world-to-camera convention from
geometry.py, synchronized integer exposure ns. Output: copied observations with
contradicted scores withdrawn and explicit evidence. No coordinates are invented.
Distances are pixels; triangulated world points are metres. Insufficient evidence
leaves labels unverified. Geometry cannot resolve a global left/right permutation.
"""
import copy
import numpy as np
from .geometry import CameraGeometry, arrays, project, triangulate


class ObservationConsistency:
    def __init__(self, cameras, config):
        self.cameras, self.cfg = cameras, config
        self.cache = {}

    def update(self, row):
        out = copy.deepcopy(row)
        names = row['joint_names']; index = {n: i for i, n in enumerate(names)}
        threshold = self.cfg['confidence_threshold']
        pixel_gate = self.cfg['ransac_threshold_px']
        views = {c: v for c, v in row['views'].items() if c in self.cameras
                 and v.get('keypoints_px') is not None
                 and abs(v['timestamp_ns']-row['timestamp_ns']) <= self.cfg['max_sync_error_ms']*1e6}
        evidence = []; independent = {}

        def other_projection(cid, name):
            key = (cid, name)
            if key in independent:
                return independent[key]
            j = index[name]
            ids = tuple(c for c, v in views.items() if c != cid and v['scores'][j] >= threshold
                        and np.isfinite(v['keypoints_px'][j]).all())
            value = None
            if len(ids) >= 2:
                cams = [self.cameras[c] for c in ids]
                if ids not in self.cache:
                    self.cache[ids] = CameraGeometry(cams)
                geom = self.cache[ids]
                fit = triangulate(cams, [views[c]['keypoints_px'][j] for c in ids],
                    [views[c]['scores'][j] for c in ids], pixel_gate,
                    max(2., self.cfg['min_triangulation_angle_deg']), geom, 'newton')
                if fit and fit['linearized_max_std_m'] < .08:
                    point = np.array(fit['xyz_m'])
                    # Do not interpret a projection behind this camera as evidence.
                    _, rotation, translation, _ = arrays(self.cameras[cid])
                    if (rotation@point+translation)[2] > 0:
                        value = project(point, self.cameras[cid])[0]
            independent[key] = value
            return value

        for cid, view in views.items():
            xy = np.asarray(view['keypoints_px'], float)
            scores = np.asarray(view['scores'], float)
            collapsed = {}; wrong = {}
            for part in ('shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle'):
                pair = [s+'_'+part for s in ('left', 'right')]
                if not all(n in index for n in pair):
                    continue
                a, b = [index[n] for n in pair]
                if min(scores[a], scores[b]) < threshold or np.linalg.norm(xy[a]-xy[b]) > 3.:
                    continue
                collapsed[part] = pair
                predictions = [other_projection(cid, n) for n in pair]
                if any(p is None for p in predictions):
                    continue
                errors = [float(np.linalg.norm(p-xy[j])) for p, j in zip(predictions, (a, b))]
                if min(errors) <= 2*pixel_gate and max(errors) > max(24., 3*pixel_gate, 4*min(errors)):
                    bad = pair[int(np.argmax(errors))]
                    wrong[part] = bad.split('_')[0]
                    out['views'][cid]['scores'][index[bad]] = 0.
                    evidence.append(dict(camera=cid, joint=bad, reason='duplicated side contradicted by other views',
                                         independent_residuals_px=errors))
            # Two independently contradicted joints anchor a collapsed limb's
            # side. A fully collapsed three-joint chain needs one independent
            # anchor: every remaining side observation is ambiguous, not repaired.
            for chain in (('shoulder', 'elbow', 'wrist'), ('hip', 'knee', 'ankle')):
                for side in ('left', 'right'):
                    anchors = sum(wrong.get(p) == side for p in chain)
                    if anchors < 2 and not (anchors == 1 and all(p in collapsed for p in chain)):
                        continue
                    for part in chain:
                        name = side+'_'+part
                        if part in collapsed and part not in wrong:
                            out['views'][cid]['scores'][index[name]] = 0.
                            evidence.append(dict(camera=cid, joint=name,
                                reason='collapsed limb with independently contradicted side anchors'))

            # Weak detections may be furniture or a different person. Test torso
            # identity against other cameras; minimum over sides tolerates swaps.
            body = [index[n] for n in names if n.split('_')[-1] in ('shoulder','elbow','wrist','hip','knee','ankle')
                    and '_hand_' not in n]
            if sum(scores[j] >= threshold for j in body) < 6:
                residuals = []
                for part in ('shoulder', 'hip'):
                    pair = [s+'_'+part for s in ('left','right')]
                    if not all(n in index for n in pair):
                        continue
                    preds = [other_projection(cid, n) for n in pair]
                    if any(p is None for p in preds):
                        continue
                    for n in pair:
                        j = index[n]
                        if scores[j] >= .2:
                            residuals.append(min(float(np.linalg.norm(p-xy[j])) for p in preds))
                if len(residuals) >= 3 and min(residuals) > max(64., 8*pixel_gate):
                    out['views'][cid]['scores'] = [0.]*len(names)
                    evidence.append(dict(camera=cid, reason='weak instance contradicted by independent torso projections',
                                         independent_residuals_px=residuals))
        out['observation_consistency'] = dict(evidence=evidence, uses_history=False,
            calibrated_probability=False, global_semantic_identity_verified=False)
        return out
