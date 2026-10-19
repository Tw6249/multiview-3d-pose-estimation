from types import SimpleNamespace
import unittest
import numpy as np
from motion_control.tracking_metrics import phase_lag,foot_contacts,standing_stop_audit


class TrackingMetricsTests(unittest.TestCase):
    def test_motionless_fallen_robot_cannot_pass_standing_stop(self):
        for height,tilt,passed in ((.79,0.,True),(.12,0.,False),(.79,1.5,False)):
            samples=[dict(root_position_m=[0,0,height],root_rpy_rad=[tilt,0,0]) for _ in range(11)]
            audit=standing_stop_audit(samples,.02)
            self.assertTrue(audit['speed_only_passed']);self.assertEqual(audit['passed'],passed)
        samples=[dict(root_position_m=[i*.01,0,.79],root_rpy_rad=[0,0,0]) for i in range(11)]
        self.assertFalse(standing_stop_audit(samples,.02)['passed'])

    def test_delay_uses_full_timeline_even_with_missing_source_interval(self):
        t=np.arange(200)*.02
        source=(np.sin(t*3)+.3*np.sin(t*7))[:,None]
        actual=np.r_[np.full((5,1),np.nan),source[:-5]]
        source[70:100]=np.nan
        result=phase_lag(actual,source,.02,(0,))['0']
        self.assertEqual(result['lag_ms'],100.)
        self.assertAlmostEqual(result['correlation'],1.)
        self.assertIsNone(phase_lag(np.zeros((100,1)),np.zeros((100,1)),.02,(0,))['0'])

    def test_simulated_slip_uses_tangent_not_vertical_contact_velocity(self):
        class Physics:
            def mj_contactForce(self,model,data,index,force):force[0]=20
            def mj_jac(self,model,data,jac,rotation_jac,position,body):jac[:]=np.eye(3)
        model=SimpleNamespace(nv=3,geom_bodyid=[0,1],body=lambda name:SimpleNamespace(id=1 if name.startswith('left') else 2))
        contact=SimpleNamespace(geom1=0,geom2=1,pos=[0,0,0],frame=[0,0,1,1,0,0,0,1,0])
        plant=SimpleNamespace(model=model,data=SimpleNamespace(contact=[contact],qvel=np.array([.3,.4,1.])),mj=Physics())
        result=foot_contacts(plant)
        self.assertEqual(result['left']['normal_force_n'],20.)
        self.assertAlmostEqual(result['left']['tangent_speeds_m_s'][0],.5)
        self.assertEqual(result['right']['tangent_speeds_m_s'],[])


if __name__=='__main__':unittest.main()
