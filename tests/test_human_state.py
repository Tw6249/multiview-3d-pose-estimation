import copy
import unittest
import numpy as np
from pose_sim.geometry import JOINTS
from pose_sim.human_state import HumanStateEstimator
from pose_sim.landmarks import HUMAN33, HUMAN33_INDICES


def observation(frame, shift=0., noise=0.):
    rng=np.random.default_rng(frame)
    p={}
    for side,y in [('left',.16),('right',-.16)]:
        for name,x,z in [('shoulder',0,1.45),('elbow',.05,1.15),('wrist',.1,.85),
                         ('hip',0,.95),('knee',.06,.55),('ankle',0,.15)]:
            p[f'{side}_{name}']=(np.array([x+shift,y,z])+rng.normal(0,noise,3)).tolist()
    q=dict(linearized_max_std_m=.005,mean_reprojection_px=1.,ray_angle_deg=25.,inlier_cameras=['a','b','c'])
    return dict(frame_id=frame,timestamp_ns=frame*40_000_000,source_id='person',landmark_schema_id='coco17',
                calibration_sha256='calibration',joint_names=JOINTS,
                joints_world_m=[p.get(n) for n in JOINTS],quality=[copy.deepcopy(q) if n in p else None for n in JOINTS])


class HumanStateTests(unittest.TestCase):
    def test_noise_reduction_preserves_translating_motion_and_bone_length(self):
        engine=HumanStateEstimator();raw_error=[];state_error=[];raw_len=[];state_len=[]
        for f in range(70):
            row=observation(f,shift=.012*f,noise=.018);truth=observation(f,shift=.012*f)
            out=engine.update(row)
            if f<20:continue
            for i in range(5,17):
                raw_error.append(np.linalg.norm(np.array(row['joints_world_m'][i])-truth['joints_world_m'][i]))
                state_error.append(np.linalg.norm(np.array(out['joints_world_m'][i])-truth['joints_world_m'][i]))
            raw_len.append(np.linalg.norm(np.array(row['joints_world_m'][13])-row['joints_world_m'][15]))
            state_len.append(np.linalg.norm(np.array(out['joints_world_m'][13])-out['joints_world_m'][15]))
        self.assertLess(np.mean(state_error),np.mean(raw_error))
        self.assertLess(np.std(state_len),np.std(raw_len)*.4)
        self.assertGreater(out['joints_world_m'][11][0],.7)

    def test_outlier_is_local_prediction_and_expires_without_fake_quality(self):
        e=HumanStateEstimator()
        for f in range(15):e.update(observation(f))
        bad=observation(15);bad['joints_world_m'][13][2]+=1.
        out=e.update(bad)
        self.assertEqual(out['joint_states'][13]['status'],'predicted')
        self.assertIsNone(out['quality'][13])
        self.assertEqual(out['joint_states'][12]['status'],'filtered')
        self.assertEqual(out['joint_states'][13]['last_observed_timestamp_ns'],14*40_000_000)
        for f in range(16,19):
            row=observation(f);row['joints_world_m'][13]=None;row['quality'][13]=None
            out=e.update(row)
        self.assertIsNone(out['joints_world_m'][13])
        self.assertEqual(out['joint_states'][13]['status'],'missing')

    def test_source_and_calibration_reset_history(self):
        for field in ('source_id','calibration_sha256','perception_signature'):
            e=HumanStateEstimator()
            for f in range(15):e.update(observation(f))
            row=observation(15,shift=3.);row[field]='new'
            out=e.update(row)
            self.assertTrue(out['reset']);self.assertEqual(out['bone_lengths_m'],{})
            self.assertAlmostEqual(out['joints_world_m'][11][0],3.)

    def test_clock_and_schema_fail_closed(self):
        e=HumanStateEstimator();e.update(observation(1))
        with self.assertRaisesRegex(ValueError,'timestamps'):e.update(observation(1))
        row=observation(2);row['timestamp_ns']=float(row['timestamp_ns'])
        with self.assertRaisesRegex(ValueError,'timestamps'):e.update(row)
        row=observation(2);row['landmark_schema_id']='robot'
        with self.assertRaisesRegex(ValueError,'schema'):e.update(row)

    def test_long_gap_does_not_continue_velocity(self):
        e=HumanStateEstimator()
        for f in range(15):e.update(observation(f,shift=.01*f))
        out=e.update(observation(100,shift=2.))
        self.assertTrue(out['reset'])
        np.testing.assert_allclose(out['velocities_world_m_s'][11],[0,0,0])

    def test_coco17_orientation_and_uncalibrated_ground_remain_unknown(self):
        out=HumanStateEstimator().update(observation(0))
        self.assertTrue(all(x is None for x in out['orientations'].values()))
        self.assertTrue(all(x is None for x in out['contacts'].values()))

    def test_contact_requires_same_point_to_be_near_ground_and_stationary(self):
        e=HumanStateEstimator({'ground_height_m':0.})
        p={n:np.array([0.,0.,z]) for n,z in [('left_heel',0.),('left_big_toe',.2),('left_small_toe',.2)]}
        for n in p:e.filters[n]=dict(x=np.r_[p[n],[1.,0,0] if n=='left_heel' else [0,0,0]],observations=3)
        result=e._contacts(p,set(p))
        self.assertLess(result['left']['stationary_contact_score'],.001)

    def test_wholebody_indices_and_foot_rotation_convention(self):
        self.assertEqual(len(HUMAN33),33);self.assertEqual(len(set(HUMAN33_INDICES)),33)
        self.assertEqual(HUMAN33_INDICES[23:28],[91,93,96,100,108])
        p={'left_heel':np.array([0.,.2,0]),'left_big_toe':np.array([.2,.18,0]),
           'left_small_toe':np.array([.2,.25,0])}
        r=np.array(HumanStateEstimator()._orientations(p,set(p))['left_foot'])
        np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-10)
        self.assertGreater(r[2,2],.99)

    def test_input_is_immutable_and_no_future_data_needed(self):
        row=observation(0);before=copy.deepcopy(row)
        a=HumanStateEstimator().update(row);b=HumanStateEstimator().update(row)
        self.assertEqual(row,before);self.assertEqual(a,b)

    def test_controller_cannot_treat_prediction_as_new_image_observation(self):
        from motion_control.humanoid import read_body
        row=observation(1);row['state_schema_id']='human_state_v1'
        row['joint_states']=[dict(status='filtered',last_observed_timestamp_ns=row['timestamp_ns']) for _ in JOINTS]
        self.assertEqual(read_body(row).shape,(12,3))
        row['joint_states'][13]={'status':'predicted','last_observed_timestamp_ns':0}
        with self.assertRaisesRegex(ValueError,'Missing left_knee'):read_body(row)

    def test_default_single_frame_confirmation_keeps_legacy_status(self):
        out=HumanStateEstimator().update(observation(0))
        self.assertEqual(out['joint_states'][13]['status'],'filtered')
        self.assertIsNone(out['joint_states'][13]['confirmation_streak'])

    def test_new_track_is_tentative_until_consecutive_acceptances(self):
        e=HumanStateEstimator({'track_confirmation_frames':3})
        seen=[e.update(observation(f))['joint_states'][13] for f in range(3)]
        self.assertEqual([s['status'] for s in seen],['tentative','tentative','filtered'])
        self.assertEqual([s['confirmation_streak'] for s in seen[:2]],[1,2])

    def test_tentative_tracks_do_not_feed_bone_samples(self):
        e=HumanStateEstimator({'track_confirmation_frames':3})
        e.update(observation(0));e.update(observation(1))
        self.assertEqual(e.samples,{})
        e.update(observation(2))
        self.assertTrue(e.samples)

    def test_miss_before_confirmation_restarts_count(self):
        e=HumanStateEstimator({'track_confirmation_frames':3})
        e.update(observation(0));e.update(observation(1))
        row=observation(2);row['joints_world_m'][13]=None;row['quality'][13]=None
        self.assertEqual(e.update(row)['joint_states'][13]['status'],'tentative')
        out=e.update(observation(3))['joint_states'][13]
        self.assertEqual((out['status'],out['confirmation_streak']),('tentative',1))

    def test_expired_track_must_be_confirmed_again(self):
        e=HumanStateEstimator({'track_confirmation_frames':3})
        for f in range(5):e.update(observation(f))
        for f in range(5,9):
            row=observation(f);row['joints_world_m'][13]=None;row['quality'][13]=None
            e.update(row)
        states=e.update(observation(9))['joint_states']
        self.assertEqual((states[13]['status'],states[12]['status']),('tentative','filtered'))

    def test_confirmation_frames_must_be_positive_integer(self):
        for bad in (0,2.,True):
            with self.assertRaises(ValueError):HumanStateEstimator({'track_confirmation_frames':bad})


if __name__=='__main__':unittest.main()
