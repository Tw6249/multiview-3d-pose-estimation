import copy
import unittest
import numpy as np
from pose_sim.foot_support import FootSupportEstimator,load_config


def packet(t=0,height=.005,speed=0.):
    p={s+'_'+n:[x,y,height] for s,y in [('left',.1),('right',-.1)]
       for n,x in [('heel',-.06),('big_toe',.16),('small_toe',.14)]}
    v={n:[speed,0,0] for n in p}
    states={n:dict(status='filtered',last_observed_timestamp_ns=t,max_std_m=.01,
        velocity_covariance_m2_s2=(np.eye(3)*.01).tolist()) for n in p}
    return [t,p,v,states,dict(left=True,right=True),0.]


class FootSupportTests(unittest.TestCase):
    def run_samples(self,e,height=.005,speed=0.,start=0):
        for i in range(start,start+5):r=e.update(*packet(i*40_000_000,height,speed))
        return r

    def test_stationary_and_whole_foot_lift_have_distinct_evidence(self):
        e=FootSupportEstimator();r=self.run_samples(e)
        self.assertEqual(r['phase'],'double');self.assertFalse(r['physical_support_verified'])
        r=e.update(*packet(200_000_000,.2))
        self.assertEqual(r['phase'],'unknown')
        r=self.run_samples(e,.2,start=6);self.assertEqual(r['phase'],'flight')

    def test_ground_slide_is_unknown_not_a_swing(self):
        r=self.run_samples(FootSupportEstimator(),speed=1.)
        self.assertEqual(r['feet']['left']['state'],'unknown')
        self.assertLess(r['feet']['left']['score'],.25)

    def test_clearance_does_not_require_velocity_observability(self):
        e=FootSupportEstimator()
        for i in range(5):
            a=packet(i*40_000_000,.2);a[2].clear()
            for s in a[3].values():s.pop('velocity_covariance_m2_s2')
            r=e.update(*a)
        self.assertEqual(r['phase'],'flight')

    def test_toe_pivot_can_support_while_heel_moves(self):
        e=FootSupportEstimator()
        for i in range(5):
            a=packet(i*40_000_000);a[1]['left_heel'][2]=.15;a[2]['left_heel']=[.5,0,.3]
            r=e.update(*a)
        self.assertEqual(r['feet']['left']['state'],'contact')
        self.assertNotIn('left_heel',r['feet']['left']['witness_landmarks'])

    def test_ground_and_stationarity_must_belong_to_same_point(self):
        e=FootSupportEstimator()
        for i in range(5):
            a=packet(i*40_000_000,.2);a[1]['left_heel'][2]=0.;a[2]['left_heel']=[1.,0,0]
            r=e.update(*a)
        self.assertEqual(r['feet']['left']['state'],'unknown')
        self.assertEqual(r['feet']['right']['state'],'swing')

    def test_ambiguous_evidence_cannot_hold_previous_contact(self):
        e=FootSupportEstimator();self.run_samples(e)
        r=e.update(*packet(200_000_000,speed=.25));self.assertEqual(r['phase'],'unknown')
        r=e.update(*packet(240_000_000));self.assertEqual(r['phase'],'unknown')

    def test_uncertain_position_velocity_and_malformed_covariance_withdraw(self):
        for field,value in [('max_std_m',.2),('velocity_covariance_m2_s2',(np.eye(3)*4).tolist()),
                            ('velocity_covariance_m2_s2',(-np.eye(3)).tolist()),('last_observed_timestamp_ns',0),('status','predicted')]:
            e=FootSupportEstimator();self.run_samples(e);a=packet(200_000_000)
            a[3]['left_big_toe'][field]=value;r=e.update(*a)
            self.assertEqual(r['feet']['left']['state'],'unknown',field)
            self.assertEqual(r['feet']['right']['state'],'contact')

    def test_plane_attachment_missing_and_gap_reset_confirmation(self):
        for mode in ('floor','attachment','point','gap'):
            e=FootSupportEstimator();self.run_samples(e);a=packet(400_000_000 if mode=='gap' else 200_000_000)
            if mode=='floor':a[-1]=None
            if mode=='attachment':a[4]['left']=False
            if mode=='point':a[1].pop('left_heel')
            self.assertEqual(e.update(*a)['feet']['left']['state'],'unknown',mode)

    def test_world_yaw_and_translation_preserve_labels(self):
        a,b=FootSupportEstimator(),FootSupportEstimator();R=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        for i in range(5):
            x=packet(i*40_000_000);y=copy.deepcopy(x)
            y[1]={n:(R@p+[3,4,5]).tolist() for n,p in y[1].items()};y[-1]=5.
            self.assertEqual(a.update(*x)['phase'],b.update(*y)['phase'])

    def test_bad_time_or_configuration_fails_closed(self):
        e=FootSupportEstimator();e.update(*packet())
        with self.assertRaises(ValueError):e.update(*packet())
        for key,value in [('confirmation_frames',1),('minimum_swing_height_m',float('nan'))]:
            cfg=load_config();cfg[key]=value
            with self.assertRaises(ValueError):FootSupportEstimator(cfg)


if __name__=='__main__':unittest.main()
