import unittest
import numpy as np
from pose_sim.support import AnkleSupportEstimator
from motion_control.contact_reference import load_config


class SupportTests(unittest.TestCase):
    def estimator(self):return AnkleSupportEstimator(load_config()['support'])

    def test_stationary_evidence_confirms_but_does_not_claim_measured_contact(self):
        e=self.estimator();p=dict(left_ankle=[0,.1,.08],right_ankle=[0,-.1,.08])
        for i in range(12):r=e.update(i*40_000_000,p,.8)
        self.assertEqual(r['feet']['left']['status'],'support_candidate')
        self.assertFalse(r['measured_contact']);self.assertFalse(r['calibrated_probability'])

    def test_low_moving_foot_is_not_locked_and_same_side_swing_is_preserved(self):
        e=self.estimator()
        for i in range(20):
            r=e.update(i*40_000_000,dict(left_ankle=[i*.03,.1,.08],right_ankle=[0,-.1,.25]),.8)
        self.assertEqual(r['feet']['left']['status'],'unknown')
        self.assertEqual(r['feet']['right']['status'],'swing')

    def test_missing_and_long_gap_clear_contact_confirmation(self):
        e=self.estimator();p=dict(left_ankle=[0,.1,.08],right_ankle=[0,-.1,.08])
        for i in range(12):e.update(i*40_000_000,p,.8)
        r=e.update(480_000_000,p,.8,current=False)
        self.assertEqual(r['feet']['left']['status'],'unknown')
        r=e.update(520_000_000,p,.8)
        self.assertEqual(r['feet']['left']['status'],'unknown')
        for i in range(14,25):e.update(i*40_000_000,p,.8)
        r=e.update(2_000_000_000,p,.8)
        self.assertEqual(r['feet']['left']['status'],'unknown')
        with self.assertRaises(ValueError):e.update(2_000_000_000,p,.8)

    def test_coordinate_translation_and_body_scale_do_not_change_support(self):
        a,b=self.estimator(),self.estimator()
        for i in range(20):
            p=dict(left_ankle=[i*.02,.1,.08],right_ankle=[0,-.1,.25])
            shifted={k:np.array(v)*1.4+[2,3,4] for k,v in p.items()}
            x=a.update(i*40_000_000,p,.8);y=b.update(i*40_000_000,shifted,1.12)
            self.assertEqual([v['status'] for v in x['feet'].values()],[v['status'] for v in y['feet'].values()])

    def test_slow_accumulated_drift_releases_a_support_hypothesis(self):
        e=self.estimator();statuses=[]
        for i in range(50):
            x=max(0,i-15)*.005
            r=e.update(i*40_000_000,dict(left_ankle=[x,.1,.08],right_ankle=[0,-.1,.08]),.8)
            statuses.append(r['feet']['left']['status'])
        self.assertEqual(statuses[14],'support_candidate')
        self.assertIn('unknown',statuses[20:35])

    def test_malformed_geometry_withdraws_instead_of_reusing_contact(self):
        for value in (None,'bad',[.8,.9]):
            e=self.estimator();p=dict(left_ankle=[0,.1,.08],right_ankle=[0,-.1,.08])
            for i in range(12):e.update(i*40_000_000,p,.8)
            r=e.update(480_000_000,p,value)
            self.assertEqual(r['feet']['left']['status'],'unknown')
            self.assertEqual(e.anchor_source,[None,None])
        r=e.update(520_000_000,dict(left_ankle=['invalid'],right_ankle=[0,-.1,.08]),.8)
        self.assertEqual(r['feet']['right']['status'],'unknown')


if __name__=='__main__':unittest.main()
