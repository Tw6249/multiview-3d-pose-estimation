import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.benchmark_twist_policy import OfflineMotion, AuditPlant, balance_violation, alignment_reference
from motion_control.twist import TwistPlant, DEFAULT_Q, DEFAULT_REF


class OfflineAdapterTests(unittest.TestCase):
    def test_frozen_alignment_does_not_repeat_motion_velocity(self):
        nominal=DEFAULT_REF.copy();nominal[[0,1,5]]=[1,.5,-16.]
        nominal[6]+=.2
        ref=alignment_reference(nominal,.5)
        np.testing.assert_array_equal(ref[[0,1,5]],0.)
        self.assertAlmostEqual(ref[6],DEFAULT_REF[6]+.1)

    def test_central_rates_are_held_at_left_sample(self):
        data=dict(fps=10.,root_pos=np.array([[0,0,.8],[.1,0,.8],[.4,0,.8]]),
                  root_rot=Rotation.from_euler('z',[0,.1,.4]).as_quat(),dof_pos=np.zeros((3,29)))
        motion=OfflineMotion(data)
        np.testing.assert_allclose(motion.velocity[:,0],[1,2,3])
        np.testing.assert_allclose(motion.omega[:,2],[1,2,3])
        reference,_,rotation=motion.sample(.15)
        self.assertAlmostEqual(reference[5],2.)
        np.testing.assert_allclose(reference[:2],rotation.inv().apply([2,0,0])[:2])

    def test_world_velocity_is_rotated_to_body_axes(self):
        data=dict(fps=10.,root_pos=np.array([[0,0,.8],[0,.1,.8],[0,.2,.8]]),
                  root_rot=Rotation.from_euler('z',[np.pi/2]*3).as_quat(),dof_pos=np.zeros((3,29)))
        reference,position,rotation=OfflineMotion(data).sample(.05)
        np.testing.assert_allclose(reference[:2],[1,0],atol=1e-12)
        np.testing.assert_allclose(position,[0,.05,.8],atol=1e-12)
        self.assertAlmostEqual(reference[5],0)

    def test_yaw_wrap_has_small_rate_and_slerp(self):
        angles=np.deg2rad([179,181,183])
        data=dict(fps=10.,root_pos=np.tile([0,0,.8],(3,1)),
                  root_rot=Rotation.from_euler('z',angles).as_quat(),dof_pos=np.zeros((3,29)))
        reference,_,rotation=OfflineMotion(data).sample(.05)
        self.assertAlmostEqual(reference[5],np.deg2rad(20),places=10)
        self.assertAlmostEqual(abs(rotation.as_euler('xyz')[2]),np.pi,places=10)

    def test_balance_diagnostic_keeps_late_failure(self):
        q=np.tile(np.r_[0,0,.8,1,0,0,0,DEFAULT_Q],(100,1))
        q[-1,2]=.3
        violation=balance_violation(q)
        self.assertEqual(np.flatnonzero(violation).tolist(),[99])

    def test_audit_does_not_change_production_dynamics(self):
        production,audit=TwistPlant(),AuditPlant()
        for i in range(8):
            target=DEFAULT_Q.copy();target[15]+=.02*np.sin(i)
            if i==3:
                production.data.xfrc_applied[production.model.body('pelvis').id,0]=10.
                audit.data.xfrc_applied[audit.model.body('pelvis').id,0]=10.
            production.step(target);audit.step(target)
            np.testing.assert_allclose(audit.data.qpos,production.data.qpos,rtol=0,atol=1e-13)
            np.testing.assert_allclose(audit.data.qvel,production.data.qvel,rtol=0,atol=1e-13)
            self.assertLessEqual(audit.records[-1]['max_torque_ratio'],1+1e-12)


if __name__=='__main__':unittest.main()
