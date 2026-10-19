"""Causal filtering and observable body axes; no extrapolated observations."""
from dataclasses import dataclass
import numpy as np
from .contracts import timestamp, vector

REQUIRED = ('left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip')


def unit(v):
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm < 1e-6:
        raise ValueError('Degenerate body segment')
    return v/norm


class OneEuro:
    def __init__(self, config):
        self.cfg = config
        self.raw = self.filtered = self.derivative = None

    def update(self, x, dt):
        if self.raw is None:
            self.raw, self.filtered, self.derivative = x.copy(), x.copy(), np.zeros_like(x)
        else:
            def alpha(cutoff):
                return 1/(1 + 1/(2*np.pi*cutoff*dt))
            ad = alpha(self.cfg['derivative_cutoff_hz'])
            self.derivative += ad*((x-self.raw)/dt - self.derivative)
            cutoff = self.cfg['min_cutoff_hz'] + self.cfg['beta']*np.linalg.norm(self.derivative, axis=-1, keepdims=True)
            self.filtered += alpha(cutoff)*(x-self.filtered)
            self.raw = x.copy()
        return self.filtered.copy()


@dataclass(frozen=True)
class BodyObservation:
    timestamp_ns: int
    frame_id: int
    origin_world_m: np.ndarray
    rotation_body_to_world: np.ndarray
    arm_directions_body: dict
    points_world_m: np.ndarray

    def to_dict(self):
        return {'timestamp_ns': self.timestamp_ns, 'frame_id': self.frame_id,
                'origin_world_m': self.origin_world_m.tolist(),
                'rotation_body_to_world': self.rotation_body_to_world.tolist(),
                'arm_directions_body': {k: v.tolist() for k, v in self.arm_directions_body.items()},
                'joint_names': REQUIRED, 'points_world_m': self.points_world_m.tolist()}


class PoseProcessor:
    def __init__(self, config):
        self.cfg, self.filter = config, OneEuro(config)
        self.last_seen = self.last_valid = None
        self.raw = self.lengths = None
        self.reason = 'no observation'

    def process(self, row):
        """Reject the entire control observation if any required point is bad.

        The source skeleton remains untouched. Returning None actively withdraws
        the target; a cached/filtered point never acquires a fresh timestamp.
        """
        ts = timestamp(row['timestamp_ns'])
        if self.last_seen is not None and ts <= self.last_seen:
            self.reason = 'nonmonotonic pose timestamp'
            return None
        self.last_seen = ts
        try:
            names = row['joint_names']
            if len(names) != len(set(names)):
                raise ValueError('Duplicate pose joint names')
            indices = [names.index(n) for n in REQUIRED]
            points = []
            for i in indices:
                p, q = row['joints_world_m'][i], row['quality'][i]
                if p is None or q is None:
                    raise ValueError('Missing required body point')
                error, angle = q['mean_reprojection_px'], q['ray_angle_deg']
                sigma = q['linearized_max_std_m']
                if not np.isfinite(sigma) or not 0 <= sigma <= self.cfg['max_linearized_std_m']:
                    raise ValueError('Triangulation uncertainty exceeds control budget')
                if (not np.isfinite([error, angle]).all() or error < 0 or
                        error > self.cfg['max_reprojection_px'] or not self.cfg['min_ray_angle_deg'] <= angle <= 90 or
                        len(set(q['inlier_cameras'])) < 2):
                    raise ValueError('Insufficient geometric quality')
                points.append(vector(p, 3, 'body point'))
            raw = np.array(points)
            lengths = np.array([np.linalg.norm(raw[b]-raw[a]) for a, b in ((0,2),(2,4),(1,3),(3,5))])
            if np.any(lengths < self.cfg['min_bone_m']) or np.any(lengths > self.cfg['max_bone_m']):
                raise ValueError('Implausible arm length')
            dt = (ts-self.last_valid)/1e9 if self.last_valid is not None else 0
            reset = self.last_valid is None or dt > self.cfg['reset_gap_s']
            if not reset:
                if np.max(np.linalg.norm(raw-self.raw, axis=1))/dt > self.cfg['max_joint_speed_m_s']:
                    raise ValueError('Body point jump')
                if np.max(np.maximum(lengths/self.lengths, self.lengths/lengths)) > self.cfg['max_bone_ratio']:
                    raise ValueError('Bone length discontinuity')
            # Validate raw body axes BEFORE advancing filter state.
            self._body_axes(raw)
            if reset:
                self.filter = OneEuro(self.cfg)
                self.lengths = lengths.copy()
            points = self.filter.update(raw, dt)
            origin, rotation = self._body_axes(points)
            directions = {side: np.array([unit(points[b]-points[a]) @ rotation,
                                           unit(points[c]-points[b]) @ rotation])
                          for side, (a,b,c) in {'left':(0,2,4), 'right':(1,3,5)}.items()}
            self.raw, self.last_valid = raw, ts
            self.reason = 'accepted'
            return BodyObservation(ts, row['frame_id'], origin, rotation, directions, points)
        except (ValueError, KeyError, IndexError, TypeError) as error:
            self.reason = str(error)
            return None

    @staticmethod
    def _body_axes(p):
        origin = (p[6]+p[7])/2
        up = unit((p[0]+p[1])/2-origin)
        left = unit(p[0]-p[1])
        forward = np.cross(left, up)
        if np.linalg.norm(forward) < 0.3 or np.linalg.norm(p[0]-p[1]) < 0.1:
            raise ValueError('Unobservable torso frame')
        forward = unit(forward)
        left = unit(np.cross(up, forward))
        return origin, np.column_stack((forward, left, up))
