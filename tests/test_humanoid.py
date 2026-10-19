import copy
import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from motion_control.humanoid_replay import ReferenceGate
from motion_control.root_tracking import RootTracker
from motion_control.twist import DEFAULT_REF,DEFAULT_Q,KP,KD
from motion_control.unitree import SDK_JOINT_NAMES


class ReferenceTests(unittest.TestCase):
    def test_expiry_invalid_and_out_of_order(self):
        gate=ReferenceGate();row=dict(timestamp_ns=10**18,accepted=True,reference=DEFAULT_REF.tolist())
        gate.receive(row,10**18+120_000_000)
        self.assertIsNotNone(gate.get(10**18+250_000_000))
        self.assertIsNone(gate.get(10**18+250_000_001))
        with self.assertRaises(ValueError):gate.receive(row,10**18+120_000_000)
        gate.receive(dict(timestamp_ns=10**18+40_000_000,accepted=False),10**18+160_000_000)
        self.assertIsNone(gate.get(10**18+160_000_000))
        with self.assertRaises(ValueError):gate.receive(dict(row,timestamp_ns=10**18+500_000_000),10**18+200_000_000)

    def test_root_path_tracks_source_translation_and_turn_not_hand(self):
        first=dict(root_position_m=[1,2,.8],root_heading_rad=np.pi/2)
        tracker=RootTracker(first,[0,0,.8],0)
        moved=dict(root_position_m=[1,2.5,.8],root_heading_rad=np.pi/2+.2)
        reference,error=tracker.correct(moved,DEFAULT_REF,[0,0,.8],0)
        self.assertGreater(reference[0],0);self.assertGreater(reference[5],0)
        np.testing.assert_allclose(error['target_position_xy_m'],[.5,0],atol=1e-9)
        np.testing.assert_array_equal(reference[6:],DEFAULT_Q)


@unittest.skipUnless(importlib.util.find_spec('mujoco'),'Requires frozen robot environment and pinned TWIST2 assets')
class HumanoidTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from motion_control.twist import TwistPlant
        cls.plant=TwistPlant()

    def body(self,q=None,scale=1):
        from motion_control.humanoid import NAMES
        p=self.plant;p.data.qpos[:7]=[0,0,.793,1,0,0,0]
        p.data.qpos[7:]=DEFAULT_Q if q is None else q;p.mj.mj_kinematics(p.model,p.data)
        mapping=dict(shoulder='shoulder_roll',elbow='elbow',wrist='wrist_yaw',hip='hip_pitch',knee='knee',ankle='ankle_pitch')
        return np.array([p.data.body(side+'_'+mapping[joint]+'_link').xpos.copy()
                         for name in NAMES for side,joint in [name.split('_',1)]])*scale

    def test_scale_invariance_and_leg_correspondence(self):
        from motion_control.humanoid import FullBodyRetargeter
        base=FullBodyRetargeter().solve(self.body(),0)
        bigger=FullBodyRetargeter().solve(self.body(scale=1.4),0)
        np.testing.assert_allclose(base['reference'],bigger['reference'],atol=.015)
        self.assertAlmostEqual(base['scale_to_target']/bigger['scale_to_target'],1.4,places=4)
        q=DEFAULT_Q.copy();q[0]=-.6;q[3]=1.1
        bent=FullBodyRetargeter().solve(self.body(q),0)
        self.assertGreater(bent['reference'][6+3]-base['reference'][6+3],.4)
        self.assertLess(abs(bent['reference'][6+9]-base['reference'][6+9]),.03)

    def test_raising_arms_does_not_trigger_walk(self):
        from motion_control.humanoid import FullBodyRetargeter
        base=FullBodyRetargeter().solve(self.body(),0)
        q=DEFAULT_Q.copy();q[15]=-.8;q[22]=-.8
        raised=FullBodyRetargeter().solve(self.body(q),0)
        np.testing.assert_allclose(raised['reference'][:2],[0,0],atol=1e-12)
        np.testing.assert_allclose(raised['reference'][6:18],base['reference'][6:18],atol=.03)
        self.assertGreater(abs(raised['reference'][21]-base['reference'][21]),.5)

    def test_adaptive_body_size_does_not_make_path_depend_on_world_origin(self):
        from motion_control.humanoid import FullBodyRetargeter
        a=FullBodyRetargeter(input_filtered=True);b=FullBodyRetargeter(input_filtered=True)
        previous=np.zeros(2)
        for i in range(20):
            body=self.body(scale=1+.003*i)+[i*.006,0,0]
            base=a.solve(body,i*40_000_000)
            shifted=b.solve(body+[20.,-15.,0.],i*40_000_000)
            np.testing.assert_allclose(base['root_position_m'],shifted['root_position_m'],atol=1e-10)
            np.testing.assert_allclose(base['reference'],shifted['reference'],atol=1e-8)
            path=np.array(base['root_position_m'][:2])
            if i:
                R=Rotation.from_euler('z',base['root_heading_rad']).as_matrix()[:2,:2]
                np.testing.assert_allclose((path-previous)/.04,R@base['reference'][:2],atol=1e-10)
            previous=path

    def test_arm_occlusion_does_not_withdraw_legs_and_rejection_rolls_back(self):
        from motion_control.humanoid import FullBodyRetargeter
        e=FullBodyRetargeter();e.solve(self.body(),0)
        missing=self.body();missing[4]=np.nan
        partial=e.solve(missing,40_000_000)
        self.assertFalse(partial['arm_observed'][0]);self.assertIsNone(partial['source_body_points_m'][4])
        before=copy.deepcopy(e.filter.filtered);q=e.q.copy()
        invalid=self.body();invalid[10]=np.nan
        with self.assertRaises(ValueError):e.solve(invalid,80_000_000)
        np.testing.assert_array_equal(e.filter.filtered,before);np.testing.assert_array_equal(e.q,q)
        self.assertEqual(e.last_time,40_000_000)

    def test_policy_history_units_order_and_ankle_mask(self):
        from motion_control.twist import TwistPolicy
        p=TwistPolicy();obs=p.observation(DEFAULT_REF,DEFAULT_Q,np.ones(29),[1,0,0,0],[1,2,3])
        self.assertEqual(obs.shape,(1,1432));np.testing.assert_allclose(obs[0,35:38],[.25,.5,.75])
        np.testing.assert_array_equal(obs[0,127:1397],0)
        velocity=obs[0,69:98];self.assertEqual(velocity[4],0);self.assertAlmostEqual(float(velocity[3]),.05)
        p.last_action=np.full(29,.3);second=p.observation(DEFAULT_REF,DEFAULT_Q,np.zeros(29),[1,0,0,0],[0,0,0])
        np.testing.assert_allclose(second[0,98:127],.3);np.testing.assert_allclose(second[0,127+9*127:1397],obs[0,:127])

    def test_motor_boundary_identity_expiry_replay_and_limits(self):
        from motion_control.g1_command import G1SimulationSink
        p=self.plant;sink=G1SimulationSink(p)
        packet=dict(hardware_enabled=False,clock_domain='offline_simulation',target_model_sha256=p.sha256,
            joint_names=list(SDK_JOINT_NAMES),sequence=0,timestamp_ns=0,valid_until_ns=60_000_000,
            position_rad=DEFAULT_Q.tolist(),kp=KP.tolist(),kd=KD.tolist(),effort_limit_nm=p.effort.tolist(),
            velocity_rad_s=[0.]*29,feedforward_torque_nm=[0.]*29)
        for change in [dict(valid_until_ns=0),dict(target_model_sha256='bad'),dict(hardware_enabled=True),
                       dict(joint_names=list(reversed(SDK_JOINT_NAMES))),dict(position_rad=[10.]*29)]:
            with self.assertRaises(ValueError):sink.validate(dict(packet,**change),0)
        sink.validate(packet,0)
        with self.assertRaises(ValueError):sink.validate(packet,1)
        with self.assertRaises(ValueError):G1SimulationSink(p).validate(packet,60_000_000)

    def test_free_base_stand_no_pelvis_override(self):
        from motion_control.twist import TwistPlant,TwistPolicy
        p=TwistPlant();policy=TwistPolicy()
        for _ in range(250):p.step(policy.step(DEFAULT_REF,*p.state()))
        self.assertEqual((p.model.nq,p.model.nv,p.model.nu),(36,35,29))
        self.assertGreater(p.data.qpos[2],.7)
        self.assertLess(np.max(np.abs(Rotation.from_quat(p.data.qpos[[4,5,6,3]]).as_euler('xyz')[:2])),.15)


if __name__=='__main__':unittest.main()
