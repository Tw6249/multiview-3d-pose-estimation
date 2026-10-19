"""G1 target description and OFFLINE SDK request serialization, no device I/O."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from .contracts import vector

ASSETS = Path(__file__).resolve().parents[1]/'assets/unitree'
SDK_JOINT_NAMES = (
    'left_hip_pitch_joint','left_hip_roll_joint','left_hip_yaw_joint','left_knee_joint','left_ankle_pitch_joint','left_ankle_roll_joint',
    'right_hip_pitch_joint','right_hip_roll_joint','right_hip_yaw_joint','right_knee_joint','right_ankle_pitch_joint','right_ankle_roll_joint',
    'waist_yaw_joint','waist_roll_joint','waist_pitch_joint',
    'left_shoulder_pitch_joint','left_shoulder_roll_joint','left_shoulder_yaw_joint','left_elbow_joint','left_wrist_roll_joint','left_wrist_pitch_joint','left_wrist_yaw_joint',
    'right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint','right_elbow_joint','right_wrist_roll_joint','right_wrist_pitch_joint','right_wrist_yaw_joint')


class G1Description:
    def __init__(self, variant='29dof'):
        if variant not in ('23dof','29dof'):
            raise ValueError('G1 variant must be explicitly 23dof or 29dof')
        self.variant = variant
        manifest = json.loads((ASSETS/'manifest.json').read_text())
        name = f'g1_{variant}.urdf'
        raw = (ASSETS/name).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        if self.sha256 != manifest['files'][name]['sha256']:
            raise ValueError('G1 URDF checksum mismatch')
        self.nodes = []
        for element in ET.fromstring(raw).findall('joint'):
            if element.attrib['type'] not in ('fixed','revolute'):
                raise ValueError('Unsupported joint type')
            origin = element.find('origin')
            xyz = np.fromstring(origin.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get('rpy','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
            transform = np.eye(4)
            transform[:3,:3] = Rotation.from_euler('xyz',rpy).as_matrix()
            transform[:3,3] = xyz
            node = {'name':element.attrib['name'],'type':element.attrib['type'],
                    'parent':element.find('parent').attrib['link'],'child':element.find('child').attrib['link'],
                    'origin':transform}
            if node['type'] == 'revolute':
                node['axis'] = np.fromstring(element.find('axis').attrib['xyz'],sep=' ')
                limit = element.find('limit').attrib
                node.update(lower=float(limit['lower']),upper=float(limit['upper']),
                            velocity=float(limit['velocity']),effort=float(limit['effort']))
            self.nodes.append(node)
        moving = {n['name']:n for n in self.nodes if n['type']=='revolute'}
        self.names = tuple(n for n in SDK_JOINT_NAMES if n in moving)
        if len(self.names) != int(variant[:2]) or set(self.names) != set(moving):
            raise ValueError('Unexpected G1 joint schema')
        self.sdk_indices = tuple(SDK_JOINT_NAMES.index(n) for n in self.names)
        self.lower = np.array([moving[n]['lower'] for n in self.names])
        self.upper = np.array([moving[n]['upper'] for n in self.names])
        self.size = len(self.names)

    def forward(self, positions):
        q = dict(zip(self.names,vector(positions,self.size,'G1 joint positions')))
        links,pending = {'pelvis':np.eye(4)},list(self.nodes)
        while pending:
            progress = False
            for node in pending[:]:
                if node['parent'] not in links: continue
                joint = node['origin'].copy()
                if node['type']=='revolute':
                    rotation = np.eye(4)
                    rotation[:3,:3] = Rotation.from_rotvec(node['axis']*q[node['name']]).as_matrix()
                    joint = joint@rotation
                links[node['child']] = links[node['parent']]@joint
                pending.remove(node); progress = True
            if not progress: raise ValueError('Disconnected/cyclic URDF')
        return links


def locomotion_request(velocity, duration_s, mode='native_locomotion'):
    """Format pinned SDK SetVelocity arguments. Not a DDS wire packet or sender.

    No motor/arm targets are accepted in this route: a learned/whole-body policy
    owns all joints when using low-level control and must not compete with sport.
    """
    if mode != 'native_locomotion':
        raise ValueError('Motor/whole-body references require a separately validated balance controller')
    velocity = vector(velocity,3,'G1 base velocity')
    if not np.isfinite(duration_s) or not 0 < duration_s <= 0.2:
        raise ValueError('Require short bounded command duration')
    if np.any(np.abs(velocity) > [0.25,0.15,0.4]):
        raise ValueError('Velocity exceeds this offline integration profile')
    return {'dry_run':True,'service':'sport','api_version':'1.0.0.0','api_id':7105,
            'sdk_method':'LocoClient.SetVelocity',
            'parameters':{'velocity':velocity.tolist(),'duration':duration_s},
            'units':['m/s','m/s','rad/s']}
