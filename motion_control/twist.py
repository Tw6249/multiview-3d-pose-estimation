"""TWIST2 full-body reference tracker; offline policy, no network/device I/O.

Adapted from amazon-far/TWIST2 b06178f, MIT (assets/humanoid/TWIST2-LICENSE.txt).
Reference: [root-local vx,vy, world height, roll,pitch, local yaw rate, 29 q].
The 29 source-derived joint references include legs; this is not velocity gait.
"""
from collections import deque
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pose_sim.robot_assets import require_asset
from .unitree import SDK_JOINT_NAMES

CONFIG=json.loads((Path(__file__).resolve().parents[1]/'config/g1_twist.json').read_text())
if tuple(CONFIG['joint_names'])!=SDK_JOINT_NAMES:raise ValueError('Policy configuration joint ordering mismatch')
DEFAULT_Q = np.array(CONFIG['default_positions_rad'])
DEFAULT_REF = np.r_[0,0,CONFIG['default_height_m'],0,0,0,DEFAULT_Q]
KP = np.array(CONFIG['kp']);KD = np.array(CONFIG['kd'])


class TwistPolicy:
    def __init__(self):
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(require_asset(CONFIG['policy_asset'])),
            sess_options=options,providers=['CPUExecutionProvider'])
        self.input=self.session.get_inputs()[0].name
        if self.session.get_inputs()[0].shape[-1] != 1432:
            raise ValueError('Wrong full-body policy observation shape')
        self.history=deque([np.zeros(127) for _ in range(10)],maxlen=10)
        self.last_action=np.zeros(29)

    def observation(self, reference, q, dq, quat_wxyz, omega_body):
        reference=np.asarray(reference,float)
        if reference.shape != (35,) or not np.isfinite(reference).all():
            raise ValueError('Require finite 35-dimensional full-body reference')
        rpy=Rotation.from_quat(np.asarray(quat_wxyz)[[1,2,3,0]]).as_euler('xyz')
        velocity=np.array(dq,copy=True);velocity[[4,5,10,11]]=0
        current=np.r_[reference,np.asarray(omega_body)*.25,rpy[:2],np.asarray(q)-DEFAULT_Q,
                      velocity*.05,self.last_action]
        result=np.r_[current,np.asarray(self.history).ravel(),reference].astype(np.float32)[None]
        self.history.append(current.copy())
        return result

    def step(self, reference, q, dq, quat_wxyz, omega_body):
        obs=self.observation(reference,q,dq,quat_wxyz,omega_body)
        action=self.session.run(None,{self.input:obs})[0].reshape(29)
        if not np.isfinite(action).all():raise ValueError('Nonfinite policy action')
        self.last_action=action.copy()
        return DEFAULT_Q+np.clip(action,-10,10)*CONFIG['action_scale']


class TwistPlant:
    """29-actuator floating-base dynamics with measured state and bounded torque."""
    def __init__(self, dt=.001):
        import mujoco
        import xml.etree.ElementTree as ET
        self.mj=mujoco
        path=require_asset(CONFIG['model_asset'])
        self.sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        for mesh in ET.parse(path).findall('.//mesh'):
            if mesh.get('file'):require_asset('twist2/assets/g1/meshes/'+mesh.get('file'))
        self.model=mujoco.MjModel.from_xml_path(str(path));self.model.opt.timestep=dt
        self.data=mujoco.MjData(self.model)
        if (self.model.nq,self.model.nv,self.model.nu)!=(36,35,29):raise ValueError('Wrong G1 dynamics model')
        self.joint_ids=np.array([self.model.joint(n).id for n in SDK_JOINT_NAMES])
        if tuple(self.model.actuator(i).name for i in range(29)) != SDK_JOINT_NAMES:
            raise ValueError('Wrong SDK motor ordering')
        self.limits=self.model.jnt_range[self.joint_ids].copy()
        # Respect BOTH upstream policy PD saturation and model motor limits.
        self.effort=np.minimum(KP,self.model.jnt_actfrcrange[self.joint_ids,1])
        self.data.qpos[:]=np.r_[0,0,.793,1,0,0,0,DEFAULT_Q]
        mujoco.mj_forward(self.model,self.data)

    def state(self):
        d=self.data
        return d.qpos[7:].copy(),d.qvel[6:].copy(),d.qpos[3:7].copy(),d.qvel[3:6].copy()

    def step(self, target, period=.02):
        target=np.clip(target,self.limits[:,0],self.limits[:,1])
        for _ in range(round(period/self.model.opt.timestep)):
            q,dq,_,_=self.state()
            self.data.ctrl[:]=np.clip(KP*(target-q)-KD*dq,-self.effort,self.effort)
            self.mj.mj_step(self.model,self.data)
        return target
