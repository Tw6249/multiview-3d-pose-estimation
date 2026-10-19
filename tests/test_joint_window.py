import copy
import unittest
import numpy as np
from pose_sim.joint_window import JointWindow,load_config
from pose_sim.pipeline import reconstruct_record
from tests.test_motion_frontend import scene


class JointWindowTests(unittest.TestCase):
    def test_causal_history_immutable_and_missing_observations_not_invented(self):
        cams,cfg,_,ds=scene();w=JointWindow(cams);original=copy.deepcopy(ds)
        for d in ds[:6]:
            raw=reconstruct_record(d,cams,cfg);saved=copy.deepcopy(raw);out=w.update(raw,d)
            self.assertEqual(raw,saved)
            self.assertEqual(out['timestamp_ns'],raw['timestamp_ns'])
            self.assertLessEqual(len(w.history),5)
        d=copy.deepcopy(ds[6])
        for v in d['views'].values():v['scores'][15]=0.
        raw=reconstruct_record(d,cams,cfg);out=w.update(raw,d)
        self.assertIsNone(out['joints_world_m'][15]);self.assertEqual(ds,original)

    def test_repeated_exposure_rejected_and_source_change_resets(self):
        cams,cfg,_,ds=scene();w=JointWindow(cams)
        for d in ds[:5]:w.update(reconstruct_record(d,cams,cfg),d)
        with self.assertRaises(ValueError):w.update(reconstruct_record(d,cams,cfg),d)
        d=copy.deepcopy(ds[5]);d['source_id']='other';out=w.update(reconstruct_record(d,cams,cfg),d)
        self.assertEqual(out['joint_window']['history_frames'],1)

    def test_current_shape_evidence_not_replaced_by_window_fit(self):
        cams,cfg,_,ds=scene();w=JointWindow(cams)
        for d in ds[:6]:
            raw=reconstruct_record(d,cams,cfg);out=w.update(raw,d)
            self.assertEqual(out['shape_observation']['joints_world_m'],raw['joints_world_m'])
            self.assertEqual(out['shape_observation']['quality'],raw['quality'])

    def test_downstream_shape_does_not_relearn_optimized_length_as_measurement(self):
        from pose_sim.human_motion import HumanMotionEstimator
        state=HumanMotionEstimator();q=dict(inlier_cameras=['a','b','c'],ray_angle_deg=30,
            linearized_max_std_m=.005,mean_reprojection_px=1.)
        for i in range(30):
            t=i*40_000_000;a=.02*i;direction=np.array([np.sin(a),0,-np.cos(a)])
            row=dict(frame_id=i,timestamp_ns=t,landmark_schema_id='coco17',joint_names=['left_hip','left_knee'],
                joints_world_m=[[0,0,1],(np.array([0,0,1])+.6*direction).tolist()],quality=[q,q])
            row['shape_observation']=dict(timestamp_ns=t,joint_names=row['joint_names'],quality=[q,q],
                joints_world_m=[[0,0,1],(np.array([0,0,1])+.4*direction).tolist()])
            out=state.update(row)
        self.assertAlmostEqual(out['shape']['left_hip--left_knee']['length_m'],.4)
        self.assertFalse(out['shape']['left_hip--left_knee']['systematic_bias_identifiable'])

    def test_constant_velocity_is_not_delayed_and_linearized_confidence_not_shrunk(self):
        from pose_sim.geometry import project
        cams,cfg,_,_=scene();w=JointWindow(cams)
        for i in range(8):
            p=np.array([.02*i,.1,1.]);d=dict(frame_id=i,timestamp_ns=i*40_000_000,joint_names=['left_wrist'],
                landmark_schema_id='coco17',source_id='s',perception_signature='m',views={c:dict(timestamp_ns=i*40_000_000,keypoints_px=project(p,cam).tolist(),scores=[.9]) for c,cam in cams.items()})
            raw=reconstruct_record(d,cams,cfg);out=w.update(raw,d)
            np.testing.assert_allclose(out['joints_world_m'][0],p,atol=1e-7)
            self.assertGreaterEqual(out['quality'][0]['linearized_max_std_m'],raw['quality'][0]['linearized_max_std_m'])

    def test_solver_cost_decreases_and_gap_clears_old_evidence(self):
        cams,cfg,_,ds=scene();w=JointWindow(cams)
        for d in ds[:6]:out=w.update(reconstruct_record(d,cams,cfg),d)
        self.assertLessEqual(out['joint_window']['objective_after'],out['joint_window']['objective_before'])
        d=copy.deepcopy(ds[6]);d['timestamp_ns']+=1_000_000_000
        for v in d['views'].values():v['timestamp_ns']=d['timestamp_ns']
        out=w.update(reconstruct_record(d,cams,cfg),d);self.assertEqual(out['joint_window']['status'],'warming')

    def test_invalid_configuration_rejected(self):
        cams,_,_,_=scene()
        for key,value in [('frames',100),('maximum_iterations',0),('pixel_scale_px',float('nan'))]:
            cfg=load_config();cfg[key]=value
            with self.assertRaises(ValueError):JointWindow(cams,cfg)


if __name__=='__main__':unittest.main()
