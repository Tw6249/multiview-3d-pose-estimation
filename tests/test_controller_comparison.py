import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from motion_control.twist import TwistPlant,DEFAULT_Q,KP,KD
from motion_control.comparison_policies import SonicAdapter,SonicV11Adapter,BeyondAdapter,Reference,rotation6,CONFIG
from scripts.benchmark_controllers import Plant,window_indices,prepare,available_index,rollout,native_window_indices


class ComparisonTests(unittest.TestCase):
    def test_direct_reference_uses_latest_sample_without_preview(self):
        for tick in range(200):
            ix=window_indices(tick,0,180)
            np.testing.assert_array_equal(ix,np.full(10,min(tick,179)))
            self.assertEqual(available_index(tick,0),tick)
        for delay in (-1,1,9):
            with self.assertRaises(ValueError):window_indices(10,delay,100)

    def test_prior_delay_protocols_are_unchanged_including_startup(self):
        for delay in (6,15):
            for tick in range(210):
                previous=np.clip(tick-delay+(np.arange(10) if delay==15 else np.zeros(10,int)),0,179)
                np.testing.assert_array_equal(window_indices(tick,delay,180),previous)

    def test_no_reference_newer_than_transport_availability(self):
        for delay in (6,15):
            for tick in range(delay,200):
                ix=window_indices(tick,delay,180)
                self.assertLessEqual(ix.max(),tick-6)
                self.assertEqual(ix[0],min(tick-delay,179))
                if delay==6:self.assertTrue(np.all(ix==ix[0]))
        np.testing.assert_array_equal(window_indices(50,15,100),np.arange(35,45))

    def test_rotation_representation_uses_rows_of_two_columns(self):
        matrix=Rotation.from_euler('xyz',[.3,-.4,.7]).as_matrix()
        expected=[matrix[0,0],matrix[0,1],matrix[1,0],matrix[1,1],matrix[2,0],matrix[2,1]]
        np.testing.assert_allclose(rotation6(matrix),expected)

    def test_instrumented_plant_preserves_twist_pd_dynamics(self):
        a,b=TwistPlant(),Plant()
        for i in range(7):
            target=DEFAULT_Q+np.sin(np.arange(29)+i)*.02
            if i==3:
                a.data.xfrc_applied[a.model.body('pelvis').id,0]=25
                b.data.xfrc_applied[b.model.body('pelvis').id,0]=25
            a.step(target);b.advance(target,KP,KD)
            np.testing.assert_allclose(a.data.qpos,b.data.qpos,rtol=0,atol=1e-13)
            np.testing.assert_allclose(a.data.qvel,b.data.qvel,rtol=0,atol=1e-13)

    def test_rejected_visual_reference_is_not_a_tracking_target(self):
        fake=dict(times=np.array([0,.04,.08]),q=np.tile(DEFAULT_Q,(3,1)),
                  xyz=np.tile([0,0,.793],(3,1)),quat=np.tile([0,0,0,1],(3,1)),valid=np.array([True,False,True]))
        fake['q'][1,:]=999
        with patch('scripts.benchmark_controllers.source',return_value=(fake,.08)):
            stream=prepare('visual')
        self.assertFalse(stream['active'][152])
        self.assertLess(np.max(np.abs(stream['bounded'][:,7:])),2)

    def test_sonic_history_mapping_and_inactive_encoder_branches(self):
        plant=Plant();policy=SonicAdapter(plant)
        ref=Reference(DEFAULT_Q.copy(),np.zeros(29),np.array([0,0,.793]),Rotation.identity(),np.zeros(3),0.)
        enc,h=policy.observations([ref]*10,plant)
        np.testing.assert_allclose(enc[4:33],DEFAULT_Q[policy.order])
        np.testing.assert_array_equal(enc[644:],0)
        self.assertEqual(len(h),930)
        np.testing.assert_allclose(h[-3:],[0,0,-1])
        # Match pinned C++ zero-entry quaternion through quat_rotate_d.
        np.testing.assert_allclose(h[-30:-3].reshape(9,3),np.tile([0,0,1],(9,1)))
        np.testing.assert_array_equal(np.arange(29)[policy.order][policy.inverse],np.arange(29))
        plant.data.qvel[3]=.25
        _,h=policy.observations([ref]*10,plant)
        self.assertEqual(h[0],0);self.assertEqual(h[27],.25)

    def test_beyond_actor_is_independent_of_embedded_motion_index(self):
        plant=Plant();policy=BeyondAdapter(plant)
        obs=np.random.default_rng(2).normal(0,.1,(1,154)).astype(np.float32)
        a=policy.policy.run(['actions'],dict(obs=obs,time_step=np.zeros((1,1),np.float32)))[0]
        b=policy.policy.run(['actions'],dict(obs=obs,time_step=np.full((1,1),100,np.float32)))[0]
        np.testing.assert_array_equal(a,b)

    def test_beyond_same_pose_has_identity_anchor_orientation(self):
        plant=Plant();policy=BeyondAdapter(plant)
        ref=Reference(plant.data.qpos[7:].copy(),np.zeros(29),plant.data.qpos[:3].copy(),Rotation.identity(),np.zeros(3),0.)
        obs=policy.observations(ref,plant)
        np.testing.assert_allclose(obs[58:64],[1,0,0,1,0,0],atol=1e-7)

    def test_beyond_config_matches_checkpoint_metadata(self):
        policy=BeyondAdapter(Plant())
        meta=policy.policy.get_modelmeta().custom_metadata_map
        self.assertEqual(meta['joint_names'].split(','),policy.cfg['policy_joint_names'])
        for name,actual in [('joint_stiffness',policy.kp),('joint_damping',policy.kd),
                            ('default_joint_pos',policy.default),('action_scale',policy.scale)]:
            expected=np.array([float(x) for x in meta[name].split(',')])[policy.inverse]
            np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)

    def test_v11_heading_observation_removes_yaw_only(self):
        plant=Plant();policy=SonicV11Adapter(plant)
        robot=Rotation.from_euler('xyz',[.3,-.2,.7])
        plant.data.qpos[3:7]=robot.as_quat()[[3,0,1,2]]
        target=Rotation.from_euler('xyz',[-.1,.25,1.2])
        ref=Reference(DEFAULT_Q.copy(),np.arange(29)*.01,np.array([0,0,.793]),target,np.zeros(3),0.)
        enc,h=policy.observations([ref]*10,plant)
        expected=rotation6((Rotation.from_euler('z',-.7)*target).as_matrix())
        np.testing.assert_allclose(enc[584:644].reshape(10,6),np.tile(expected,(10,1)),atol=1e-7)
        self.assertGreater(np.linalg.norm(expected-rotation6((robot.inv()*target).as_matrix())),.1)
        self.assertEqual(enc.shape,(1751,));self.assertEqual(h.shape,(930,))
        np.testing.assert_array_equal(enc[644:],0)
        np.testing.assert_allclose(enc[294:323],ref.dq[policy.order],atol=1e-7)

    def test_v11_official_gain_change_does_not_change_action_normalization(self):
        low=CONFIG['sonic'];new=CONFIG['sonic_v11'];scale=np.ones(29);scale[[4,10]]=1.5
        for field in ('kp','kd'):np.testing.assert_allclose(new[field],np.array(low[field])*scale)
        for field in ('default_q','action_scale','policy_joint_names'):self.assertEqual(new[field],low[field])
        self.assertEqual(new['reference_step'],5)

    def test_v11_cannot_mistake_180ms_window_for_native_900ms_preview(self):
        with self.assertRaisesRegex(ValueError,'900 ms'):
            rollout('sonic_v11','standing',{},300)

    def test_native_window_spacing_and_matched_hold_anchor(self):
        np.testing.assert_array_equal(native_window_indices('sonic_v11',100,300),np.arange(55,101,5))
        np.testing.assert_array_equal(native_window_indices('sonic',100,300),np.arange(55,65))
        for name in ('twist','beyond'):
            np.testing.assert_array_equal(native_window_indices(name,100,300),np.full(10,55))
        np.testing.assert_array_equal(native_window_indices('sonic_v11',100,300,'hold'),np.full(10,55))
        with self.assertRaises(ValueError):rollout('twist','standing',{},0,reference_mode='hold')

    def test_native_preview_never_reads_unreceived_samples_including_padding(self):
        for length in (1,8,100):
            for tick in range(length+45):
                for name in ('twist','sonic','sonic_v11','beyond'):
                    for mode in ('native','hold'):
                        ix=native_window_indices(name,tick,length,mode)
                        self.assertLessEqual(ix.max(),tick)
                        self.assertLess(ix.max(),length)
                        self.assertGreaterEqual(ix.min(),0)
                        self.assertEqual(ix[0],max(0,min(tick-45,length-1)))
        for args in (('unknown',0,1),('sonic_v11',-1,1),('sonic_v11',0,0)):
            with self.assertRaises(ValueError):native_window_indices(*args)

    def test_v11_encoder_retains_distinct_time_samples_in_native_window(self):
        plant=Plant();policy=SonicV11Adapter(plant)
        indices=native_window_indices('sonic_v11',100,300)
        window=[Reference(DEFAULT_Q+index*.001,np.full(29,index*.002),
                          np.array([0,0,.793]),Rotation.from_euler('z',index*.001),
                          np.zeros(3),.05) for index in indices]
        enc,_=policy.observations(window,plant)
        np.testing.assert_allclose(enc[4:294].reshape(10,29),
                                   [ref.q[policy.order] for ref in window],atol=1e-7)
        np.testing.assert_allclose(enc[294:584].reshape(10,29),
                                   [ref.dq[policy.order] for ref in window],atol=1e-7)
        self.assertGreater(float(np.linalg.norm(enc[4:33]-enc[265:294])),.1)
        np.testing.assert_array_equal(enc[644:],0)


if __name__=='__main__':unittest.main()
