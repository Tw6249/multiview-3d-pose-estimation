"""Configurable revolute serial chains, FK and explicit robot limits."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .contracts import vector


class RobotModel:
    def __init__(self, config):
        self.config = config
        if config['schema_version'] != 1 or config['units'] != 'SI':
            raise ValueError('Unsupported robot schema/units')
        self.robot_id = config['robot_id']
        self.sha256 = hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()
        self.joints = config['joints']
        self.names = tuple(j['name'] for j in self.joints)
        self.size = len(self.names)
        if not self.size or len(set(self.names)) != self.size:
            raise ValueError('Robot joint names must be nonempty and unique')
        self.lower = vector([j['lower_rad'] for j in self.joints], self.size, 'lower')
        self.upper = vector([j['upper_rad'] for j in self.joints], self.size, 'upper')
        self.home = vector([j['home_rad'] for j in self.joints], self.size, 'home')
        self.velocity = vector([j['max_velocity_rad_s'] for j in self.joints], self.size, 'velocity')
        self.acceleration = vector([j['max_acceleration_rad_s2'] for j in self.joints], self.size, 'acceleration')
        if (np.any(self.lower >= self.upper) or np.any(self.home < self.lower) or
                np.any(self.home > self.upper) or np.any(self.velocity <= 0) or np.any(self.acceleration <= 0)):
            raise ValueError('Invalid joint limits/home')
        self.chains = config['chains']
        used = []
        for chain in self.chains:
            vector(chain['base_m'], 3, 'base')
            indices = chain['joint_indices']
            if (len(indices) != len(chain['axes']) or len(indices) != len(chain['links_m']) or
                    len(indices) < 2 or any(type(i) is not int or not 0 <= i < self.size for i in indices)):
                raise ValueError('Malformed chain')
            for axis, link in zip(chain['axes'], chain['links_m']):
                if not np.isclose(np.linalg.norm(vector(axis, 3, 'axis')), 1):
                    raise ValueError('Joint axis must be unit length')
                vector(link, 3, 'link')
            if not 0 <= chain['elbow_after_joint'] < len(indices)-1:
                raise ValueError('Invalid elbow index')
            used.extend(indices)
        if sorted(used) != list(range(self.size)):
            raise ValueError('Each joint must belong to exactly one chain')
        if sorted(c['side'] for c in self.chains) != ['left', 'right']:
            raise ValueError('This retargeter requires one left and one right arm chain')
        for section in ('perception', 'retarget', 'control'):
            for name, value in config[section].items():
                zero_allowed = name in ('beta', 'continuity_weight')
                if (not isinstance(value, (int, float)) or not np.isfinite(value) or
                        value < 0 or (value == 0 and not zero_allowed)):
                    raise ValueError(f'{section}.{name} has an invalid range')
        control = config['control']
        if (type(control['recovery_frames']) is not int or type(config['retarget']['max_nfev']) is not int or control['rate_hz'] > 1000 or
                control['command_ttl_ms'] <= 1000/control['rate_hz'] or
                not 1000/control['rate_hz'] <= control['max_tick_interval_ms'] < control['command_ttl_ms'] or
                config['perception']['min_bone_m'] >= config['perception']['max_bone_m'] or
                config['perception']['max_bone_ratio'] <= 1):
            raise ValueError('Inconsistent control/perception configuration')
        self.forward(self.home)

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))

    def forward(self, q):
        q = vector(q, self.size, 'joint positions')
        arms = {}
        for chain in self.chains:
            base = np.array(chain['base_m'], dtype=float)
            pos, rot = base.copy(), np.eye(3)
            elbow = None
            for k, (i, axis, link) in enumerate(zip(chain['joint_indices'], chain['axes'], chain['links_m'])):
                rot = rot @ Rotation.from_rotvec(np.asarray(axis)*q[i]).as_matrix()
                pos = pos + rot @ link
                if k == chain['elbow_after_joint']:
                    elbow = pos.copy()
            upper, fore = elbow-base, pos-elbow
            lengths = np.array([np.linalg.norm(upper), np.linalg.norm(fore)])
            if np.min(lengths) < 1e-6:
                raise ValueError('Robot arm segments must have nonzero lengths')
            arms[chain['side']] = {'shoulder': base, 'elbow': elbow, 'wrist': pos,
                                   'directions': np.array([upper/lengths[0], fore/lengths[1]])}
        return arms
