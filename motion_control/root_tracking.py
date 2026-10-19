"""Source-heading velocity/path -> robot-heading reference (m, rad, m/s).

Source and target world XY are aligned at the first accepted pelvis/heading.
Positions use the aligned world; velocities must be rotated again into the
CURRENT robot heading. Source exposure time is owned/checked by ReferenceGate.
No gait or source timestamp is generated. Physical odometry is not implemented.
"""
import copy
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def wrap(angle):
    return np.arctan2(np.sin(angle),np.cos(angle))


def load_config():
    return json.loads((Path(__file__).resolve().parents[1]/'config/g1_root_tracking.json').read_text())


def planar_rotation(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s],[s,c]])


class RootTracker:
    def __init__(self,first_row,robot_position,robot_heading,config=None):
        self.cfg=copy.deepcopy(config or load_config())
        if self.cfg.get('hardware_enabled') is not False or self.cfg.get('target')!='unitree_g1_29dof':raise ValueError('Simulation G1 root profile required')
        if self.cfg['velocity_frame'] not in ('source_heading','legacy_unrotated'):raise ValueError('Explicit reference frame required')
        for key in ('position_gain_s_inv','maximum_position_feedback_axis_m_s','maximum_velocity_axis_m_s','heading_gain_s_inv','maximum_heading_feedback_rad_s'):
            if not np.isfinite(self.cfg[key]) or self.cfg[key]<=0:raise ValueError('Invalid root tracking bound')
        self.source_origin=np.asarray(first_row['root_position_m'],float)[:2].copy()
        self.source_heading=first_row['root_heading_rad']
        self.origin=np.asarray(robot_position)[:2].copy()
        self.heading=robot_heading
        self.alignment=Rotation.from_euler('z',robot_heading-self.source_heading).as_matrix()[:2,:2]
        self.evaluation_origin=np.asarray(first_row.get('source_root_position_m',first_row['root_position_m']))[:2]
        self.evaluation_alignment=Rotation.from_euler('z',robot_heading-first_row.get('source_root_heading_rad',first_row['root_heading_rad'])).as_matrix()[:2,:2]
        if not np.isfinite(np.r_[self.source_origin,self.source_heading,self.origin,self.heading,self.evaluation_origin]).all():raise ValueError('Nonfinite root alignment')

    def correct(self,row,reference,robot_position,robot_heading):
        reference=np.asarray(reference,float);robot_position=np.asarray(robot_position,float)
        if reference.shape!=(35,) or robot_position.shape!=(3,) or not np.isfinite(np.r_[reference,robot_position,robot_heading,row['root_position_m'],row['root_heading_rad']]).all():raise ValueError('Invalid root tracking input')
        target=self.origin+self.alignment@(np.asarray(row['root_position_m'])[:2]-self.source_origin)
        error=target-np.asarray(robot_position)[:2]
        target_heading=self.heading+wrap(row['root_heading_rad']-self.source_heading)
        heading_error=wrap(target_heading-robot_heading)
        R=Rotation.from_euler('z',robot_heading).as_matrix()[:2,:2]
        corrected=np.array(reference,copy=True)
        world_feedforward=planar_rotation(target_heading)@reference[:2]
        if self.cfg['velocity_frame']=='source_heading':corrected[:2]=R.T@world_feedforward
        local_feedforward=corrected[:2].copy()
        limit=self.cfg['maximum_position_feedback_axis_m_s'];speed=self.cfg['maximum_velocity_axis_m_s'];turn=self.cfg['maximum_heading_feedback_rad_s']
        corrected[:2]+=np.clip(self.cfg['position_gain_s_inv']*(R.T@error),-limit,limit)
        corrected[:2]=np.clip(corrected[:2],-speed,speed)
        corrected[5]+=np.clip(self.cfg['heading_gain_s_inv']*heading_error,-turn,turn)
        evaluation_target=self.origin+self.evaluation_alignment@(np.asarray(row.get('source_root_position_m',row['root_position_m']))[:2]-self.evaluation_origin)
        return corrected,dict(target_position_xy_m=target.tolist(),position_error_m=float(np.linalg.norm(error)),
                              source_position_error_m=float(np.linalg.norm(evaluation_target-np.asarray(robot_position)[:2])),
                              heading_error_rad=float(heading_error),velocity_frame=self.cfg['velocity_frame'],
                              source_feedforward_world_m_s=world_feedforward.tolist(),
                              applied_feedforward_world_m_s=(R@local_feedforward).tolist(),
                              feedforward_world_frame_error_m_s=float(np.linalg.norm(R@local_feedforward-world_feedforward)))
