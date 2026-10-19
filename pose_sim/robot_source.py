"""Known natural landmarks -> source rigid-link poses; no image detector or target.

This is a geometric back end, NOT a recognizer for unknown robot appearances.
Landmarks are labelled points fixed in a source link, with measured/CAD positions.
Symmetric/ambiguous features must not be assigned a unique orientation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .pipeline import read_jsonl, write_jsonl


class RobotSource:
    def __init__(self, config):
        self.config = config
        if (config.get('schema_version') != 1 or config.get('units') != 'metre' or
                config.get('perception_mode') != 'markerless'):
            raise ValueError('Require a markerless source profile in metres, schema_version=1')
        for field in ('source_id', 'landmark_schema_id'):
            if not isinstance(config.get(field), str) or not config[field].strip():
                raise ValueError(f'Source profile incomplete: supply actual {field}; see docs/source_robot_to_g1.md')
        for field in ('fit_max_rms_m', 'max_landmark_std_m', 'max_orientation_std_rad', 'min_spread_ratio'):
            if not np.isfinite(config[field]) or config[field] <= 0:
                raise ValueError('Source thresholds must be finite and positive')
        if config['min_spread_ratio'] >= 1:
            raise ValueError('min_spread_ratio must be below 1')
        self.links = config['links']
        if not isinstance(self.links, list) or not self.links:
            raise ValueError('Source profile incomplete: define actual links and natural landmarks')
        self.names, self.local_points, self.indices = [], {}, {}
        for link in self.links:
            name = link['name']
            if not isinstance(name, str) or not name or '/' in name or name in self.local_points:
                raise ValueError('Require unique nonempty link names without /')
            if type(link.get('orientation_identifiable')) is not bool:
                raise ValueError('Explicitly declare orientation_identifiable for each link')
            landmarks = link['landmarks']
            if not isinstance(landmarks, list) or not landmarks:
                raise ValueError('Each link needs named natural landmarks')
            first = len(self.names)
            points = []
            for landmark in landmarks:
                label = landmark['name']
                if not isinstance(label, str) or not label or '/' in label:
                    raise ValueError('Landmark names must be nonempty and contain no /')
                full_name = f'{name}/{label}'
                if full_name in self.names or landmark.get('feature_type') != 'natural_geometry':
                    raise ValueError('Require unique natural_geometry landmarks; artificial markers are not supported')
                point = np.asarray(landmark['position_link_m'], float)
                if point.shape != (3,) or not np.isfinite(point).all():
                    raise ValueError('Landmark position must be a finite link-local 3-vector')
                self.names.append(full_name)
                points.append(point)
            self.local_points[name] = np.array(points)
            if len(np.unique(self.local_points[name], axis=0)) != len(points):
                raise ValueError('Distinct landmarks cannot share the same link-local position')
            self.indices[name] = list(range(first, len(self.names)))
        self.sha256 = hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def fit_links(self, pose):
        if (pose.get('source_id') != self.config['source_id'] or pose['joint_names'] != self.names or
                pose.get('landmark_schema_id') != self.config['landmark_schema_id'] or
                pose.get('source_model_sha256') != self.sha256 or pose.get('perception_mode') != 'markerless'):
            raise ValueError('3D source identity/schema/model/perception mismatch')
        if not pose.get('calibration_sha256'):
            raise ValueError('Source observations must be bound to camera calibration')
        if len(pose['joints_world_m']) != len(self.names) or len(pose['quality']) != len(self.names):
            raise ValueError('3D observation count differs from source schema')
        links = {}
        for link in self.links:
            name = link['name']
            if not link['orientation_identifiable']:
                links[name] = {'valid': False, 'orientation_observed': False,
                               'reason': 'source geometry/feature identity has unresolved symmetry'}
                continue
            world, indices, stds = [], [], []
            for j, i in enumerate(self.indices[name]):
                p, quality = pose['joints_world_m'][i], pose['quality'][i]
                if p is None or not isinstance(quality, dict):
                    continue
                std = quality.get('linearized_max_std_m', np.inf)
                if not np.isfinite(std) or not 0 <= std <= self.config['max_landmark_std_m']:
                    continue
                point = np.asarray(p, float)
                if point.shape == (3,) and np.isfinite(point).all():
                    world.append(point)
                    indices.append(j)
                    stds.append(std)
            if len(world) < 3:
                links[name] = {'valid': False, 'orientation_observed': False,
                               'reason': 'fewer than three reliable natural landmarks'}
                continue
            local, world = self.local_points[name][indices], np.array(world)
            a, b = local-local.mean(axis=0), world-world.mean(axis=0)
            # Three labelled noncollinear points suffice; planar geometry is valid.
            # Check relative spread in BOTH frames, rather than a unit-dependent rank.
            spread = [np.linalg.svd(x, compute_uv=False) for x in (a, b)]
            if any(s[0] <= 1e-8 or s[1]/s[0] < self.config['min_spread_ratio'] for s in spread):
                links[name] = {'valid': False, 'orientation_observed': False,
                               'reason': 'collinear or ill-conditioned natural landmarks'}
                continue
            # Linearized rotation sensitivity with common isotropic sigma equal
            # to the largest point std. Assumes independent point errors and a
            # known exact shape; calibration/association/shape errors excluded.
            information = sum(np.dot(p, p)*np.eye(3)-np.outer(p, p) for p in a)
            rotation_std = max(stds)/np.sqrt(max(np.linalg.eigvalsh(information)[0], 1e-20))
            if rotation_std > self.config['max_orientation_std_rad']:
                links[name] = {'valid': False, 'orientation_observed': False,
                               'reason': 'landmark spacing too small relative to position uncertainty',
                               'linearized_max_orientation_std_rad': float(rotation_std)}
                continue
            u, _, vt = np.linalg.svd(a.T@b)
            rotation = vt.T@np.diag([1, 1, np.linalg.det(vt.T@u.T)])@u.T
            translation = world.mean(axis=0)-rotation@local.mean(axis=0)
            rms = float(np.sqrt(np.mean(np.sum((local@rotation.T+translation-world)**2, axis=1))))
            if rms > self.config['fit_max_rms_m']:
                links[name] = {'valid': False, 'orientation_observed': False,
                               'reason': 'rigid-link fitting residual', 'rms_m': rms}
                continue
            transform = np.eye(4)
            transform[:3, :3], transform[:3, 3] = rotation, translation
            links[name] = {'valid': True, 'T_world_link': transform.tolist(), 'fit_rms_m': rms,
                           'linearized_max_orientation_std_rad': float(rotation_std),
                           'landmark_count': len(world), 'orientation_observed': True}
        return {'schema_version': 1, 'source_id': self.config['source_id'],
                'source_model_sha256': self.sha256, 'perception_mode': 'markerless',
                'detector_model_sha256': pose.get('detector_model_sha256'),
                'frame_id': pose['frame_id'], 'timestamp_ns': pose['timestamp_ns'],
                'calibration_sha256': pose['calibration_sha256'], 'world_frame': 'calibration_world',
                'units': 'SI', 'links': links}


def fit_session(session, source):
    session = Path(session)
    rows = [source.fit_links(row) for row in read_jsonl(session/'poses3d.jsonl')]
    if not rows:
        raise ValueError('Empty source observations')
    write_jsonl(session/'link_poses.jsonl', rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['check-source', 'fit-links'])
    parser.add_argument('--session', type=Path)
    parser.add_argument('--source-config', type=Path, required=True)
    args = parser.parse_args()
    try:
        source = RobotSource(json.loads(args.source_config.read_text(encoding='utf-8')))
    except (ValueError, KeyError, TypeError) as error:
        parser.exit(2, f'SOURCE NOT READY: {error}\n')
    if args.stage == 'check-source':
        print(json.dumps({'source_profile_valid': True, 'source_id': source.config['source_id'],
                          'source_model_sha256': source.sha256, 'landmark_count': len(source.names),
                          'note': 'Geometry validation only; no image detector or physical validation implied.'}))
    elif args.session is None:
        parser.error('fit-links requires --session')
    else:
        fit_session(args.session, source)
