"""Simulation-only adapters for pinned TWIST2, SONIC and BeyondMimic models.

Input: SDK-ordered G1 reference q [rad], dq [rad/s], root XYZ [m], root
rotation [XYZW], world root velocity [m/s], yaw rate [rad/s]; fixed 20 ms ticks.
SONIC receives ten samples, supplied by an external causal scheduler. Native
offsets are 0..180 ms for low_latency, 0..900 ms for v1.1's G1 encoder.
Direct mode repeats the current sample; never query future source observations.
Output: SDK-ordered 29 joint position targets [rad], for a bounded PD plant.
Invalid/nonfinite input or output raises; no hardware transport is implemented.

SONIC conventions ported from NVlabs/GR00T-WholeBodyControl 45c03fac (Apache-2.0),
g1_deploy_onnx_ref.cpp and policy_parameters.hpp. BeyondMimic conventions from
HybridRobotics/motion_tracking_controller 742924f1 and RoboJuDo 70144b3c.
Licenses and asset provenance: assets/humanoid/controller_comparison/.
"""
from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from pose_sim.robot_assets import cache_root
from .twist import TwistPolicy, DEFAULT_REF
from .unitree import SDK_JOINT_NAMES
from .root_tracking import load_config as root_config, wrap

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT/'config/g1_controller_comparison.json').read_text())


def asset(relative):
    entries = json.loads((ROOT/'assets/humanoid/controller_comparison_manifest.json').read_text())['files']
    entry = next(x for x in entries if x['path'] == relative)
    path = cache_root()/relative
    with path.open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != entry['sha256']:
            raise ValueError('Comparison asset fingerprint mismatch: '+relative)
    return path


def session(path, width):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    engine = ort.InferenceSession(str(path), sess_options=opts, providers=['CPUExecutionProvider'])
    if engine.get_inputs()[0].shape != [1, width]:
        raise ValueError('Unexpected observation layout')
    return engine


@dataclass
class Reference:
    q: np.ndarray
    dq: np.ndarray
    xyz: np.ndarray
    rotation: Rotation
    velocity: np.ndarray
    yaw_rate: float

    def validate(self):
        if self.q.shape != (29,) or self.dq.shape != (29,) or self.xyz.shape != (3,):
            raise ValueError('Invalid reference dimensions')
        if not np.isfinite(np.r_[self.q, self.dq, self.xyz, self.rotation.as_quat(), self.velocity, self.yaw_rate]).all():
            raise ValueError('Nonfinite reference')


def rotation6(matrix):
    """First two matrix columns, row-major flattening (not column stacking)."""
    return np.asarray(matrix)[..., :, :2].reshape(-1)


class Adapter:
    def __init__(self, name):
        self.name = name
        self.cfg = CONFIG[name]
        self.kp = np.array(self.cfg['kp'])
        self.kd = np.array(self.cfg['kd'])
        self.default = np.array(self.cfg['default_q'])
        if name != 'twist':
            self.order = np.array([SDK_JOINT_NAMES.index(n) for n in self.cfg['policy_joint_names']])
            if sorted(self.order) != list(range(29)):
                raise ValueError('Joint mapping is not bijective')
            self.inverse = np.argsort(self.order)
            self.scale = np.array(self.cfg['action_scale'])
        self.last = np.zeros(29)

    def checked(self, target):
        if target.shape != (29,) or not np.isfinite(target).all():
            raise ValueError('Invalid policy target; terminate trial')
        return target


class TwistAdapter(Adapter):
    def __init__(self, plant):
        super().__init__('twist')
        self.policy = TwistPolicy()
        self.root = root_config()

    def step(self, window, plant):
        ref = window[0]; ref.validate()
        actual = Rotation.from_quat(plant.data.qpos[[4,5,6,3]])
        yaw = actual.as_euler('xyz')[2]
        heading = Rotation.from_euler('z', yaw)
        local_v = heading.inv().apply(ref.velocity)
        local_error = heading.inv().apply(ref.xyz-plant.data.qpos[:3])
        cfg = self.root
        feedback = np.clip(cfg['position_gain_s_inv']*local_error[:2],
                           -cfg['maximum_position_feedback_axis_m_s'], cfg['maximum_position_feedback_axis_m_s'])
        v = np.clip(local_v[:2]+feedback, -cfg['maximum_velocity_axis_m_s'], cfg['maximum_velocity_axis_m_s'])
        rpy = ref.rotation.as_euler('xyz')
        omega = ref.yaw_rate+np.clip(cfg['heading_gain_s_inv']*wrap(rpy[2]-yaw),
                                   -cfg['maximum_heading_feedback_rad_s'], cfg['maximum_heading_feedback_rad_s'])
        command = np.r_[v, ref.xyz[2], rpy[:2], omega, ref.q]
        return self.checked(self.policy.step(command, *plant.state()))


class SonicAdapter(Adapter):
    def __init__(self, plant, name='sonic'):
        super().__init__(name)
        self.encoder_width = self.cfg.get('encoder_width',1247)
        self.encoder = session(asset(self.cfg['encoder_asset']), self.encoder_width)
        self.decoder = session(asset(self.cfg['decoder_asset']), 994)
        self.history = deque(maxlen=10)

    def observations(self, window, plant):
        if len(window) != 10:
            raise ValueError('SONIC requires ten scheduled reference samples')
        for ref in window: ref.validate()
        q, dq, quat, omega = plant.state()
        rotation = Rotation.from_quat(quat[[1,2,3,0]])
        current = (omega.copy(), (q-self.default)[self.order], dq[self.order],
                   self.last.copy(), rotation.inv().apply([0.,0.,-1.]))
        if not self.history:
            # Official StateLogger::makeZeroEntry_ uses a zero quaternion.
            # Its quat_rotate_d(q, -Z) formula consequently returns +Z for
            # missing history (not the -Z of a valid identity rotation).
            empty=(np.zeros(3),np.zeros(29),np.zeros(29),np.zeros(29),np.array([0.,0.,1.]))
            self.history.extend([empty]*9)
        self.history.append(current)
        encoder = np.zeros(self.encoder_width, np.float32)
        # Mode 0 selects the G1 encoder. Unused teleop/SMPL branches are zero.
        encoder[4:294] = np.array([r.q[self.order] for r in window]).ravel()
        encoder[294:584] = np.array([r.dq[self.order] for r in window]).ravel()
        anchor = rotation
        if self.cfg.get('orientation_frame','body') == 'heading':
            # Match official calc_heading_quat: heading of rotated forward +X.
            forward=rotation.apply([1.,0.,0.])
            anchor=Rotation.from_euler('z',np.arctan2(forward[1],forward[0]))
        encoder[584:644] = np.concatenate([rotation6((anchor.inv()*r.rotation).as_matrix()) for r in window])
        history = np.concatenate([np.concatenate([row[j] for row in self.history]) for j in range(5)])
        return encoder, history

    def step(self, window, plant):
        enc, history = self.observations(window, plant)
        token = self.encoder.run(['encoded_tokens'], {'obs_dict':enc[None]})[0].reshape(64)
        obs = np.r_[token, history].astype(np.float32)[None]
        action = self.decoder.run(['action'], {'obs_dict':obs})[0].reshape(29)
        self.last = action.copy()
        return self.checked(self.default+self.scale*action[self.inverse])


class SonicV11Adapter(SonicAdapter):
    def __init__(self, plant):
        super().__init__(plant, name='sonic_v11')


class BeyondAdapter(Adapter):
    """RoboJuDo's motion-specific Dance_wose checkpoint; OOD on other clips."""
    def __init__(self, plant):
        super().__init__('beyond')
        self.policy = session(asset(self.cfg['policy_asset']), 154)
        self.target_data = plant.mj.MjData(plant.model)
        self.actual_data = plant.mj.MjData(plant.model)
        self.anchor = plant.model.body(self.cfg['anchor_body']).id
        md = self.policy.get_modelmeta().custom_metadata_map
        if md['observation_names'] != 'command,motion_anchor_ori_b,base_ang_vel,joint_pos,joint_vel,actions':
            raise ValueError('Unsupported BeyondMimic observation layout')

    def observations(self, ref, plant):
        ref.validate()
        self.target_data.qpos[:] = np.r_[ref.xyz, ref.rotation.as_quat()[[3,0,1,2]], ref.q]
        self.actual_data.qpos[:] = plant.data.qpos
        plant.mj.mj_kinematics(plant.model, self.target_data)
        plant.mj.mj_kinematics(plant.model, self.actual_data)
        actual = self.actual_data.xmat[self.anchor].reshape(3,3)
        desired = self.target_data.xmat[self.anchor].reshape(3,3)
        q,dq,_,omega = plant.state()
        return np.r_[ref.q[self.order], ref.dq[self.order], rotation6(actual.T@desired),
                     omega, (q-self.default)[self.order], dq[self.order], self.last].astype(np.float32)

    def step(self, window, plant):
        obs = self.observations(window[0], plant)
        # time_step only indexes the embedded motion outputs. External references
        # enter obs; regression checks action independence from time_step.
        action = self.policy.run(['actions'], {'obs':obs[None], 'time_step':np.zeros((1,1),np.float32)})[0].reshape(29)
        self.last = action.copy()
        return self.checked(self.default+self.scale*action[self.inverse])


FACTORIES = {'twist':TwistAdapter, 'sonic':SonicAdapter, 'beyond':BeyondAdapter, 'sonic_v11':SonicV11Adapter}
