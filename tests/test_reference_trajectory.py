import importlib.util
import unittest
import numpy as np
from motion_control.reference_trajectory import SourceReferenceDerivative,BoundedReferenceTrajectory,load_config
from motion_control.twist import DEFAULT_REF


class SourceDerivativeTests(unittest.TestCase):
    def test_history_uses_exposure_time_and_repeated_delivery_does_not_add_evidence(self):
        estimate=SourceReferenceDerivative()
        for i in range(3):
            ref=DEFAULT_REF.copy();ref[6]=.1*i
            row=dict(accepted=True,timestamp_ns=i*40_000_000,reference=ref.tolist(),arm_observed=[True,True])
            rate=estimate.update(row)
        self.assertAlmostEqual(rate[6],2.5)
        np.testing.assert_allclose(estimate.update(row),rate)
        self.assertEqual(len(estimate.history),3)
        estimate.update(None)
        np.testing.assert_array_equal(estimate.update(dict(row,timestamp_ns=120_000_000)),0)

    def test_large_gap_and_out_of_order_cannot_make_fresh_motion(self):
        estimate=SourceReferenceDerivative();row=dict(accepted=True,timestamp_ns=0,reference=DEFAULT_REF.tolist())
        estimate.update(row);estimate.update(dict(row,timestamp_ns=40_000_000))
        np.testing.assert_array_equal(estimate.update(dict(row,timestamp_ns=1_000_000_000)),0)
        with self.assertRaises(ValueError):estimate.update(row)


@unittest.skipUnless(importlib.util.find_spec('ruckig'),'Requires pinned robot trajectory environment')
class BoundedTrajectoryTests(unittest.TestCase):
    def engine(self):return BoundedReferenceTrajectory(DEFAULT_REF,np.tile([-2.,2.],(29,1)),'test_model')

    def test_step_reversal_and_stop_obey_all_derivative_limits(self):
        e=self.engine();target=DEFAULT_REF.copy();target[6:]=1.;previous=e.acceleration.copy()
        for i in range(180):
            if i==40:target[6:]=-1.
            if i==70:target=DEFAULT_REF.copy()
            p,d=e.step(target)
            self.assertTrue(np.all(np.abs(e.rate)<=e.max_rate+1e-7))
            self.assertTrue(np.all(np.abs(e.acceleration)<=e.max_acceleration+1e-7))
            self.assertTrue(np.all(np.abs(e.acceleration-previous)/e.dt<=e.max_jerk+1e-5))
            self.assertTrue(np.all(p>=e.lower-1e-8) and np.all(p<=e.upper+1e-8));previous=e.acceleration.copy()
        np.testing.assert_allclose(p,DEFAULT_REF,atol=1e-7)
        np.testing.assert_allclose(e.rate,0,atol=1e-7)

    def test_position_limit_projection_does_not_clip_generated_trajectory(self):
        e=self.engine();target=DEFAULT_REF.copy();target[6]=10.;rate=np.zeros(35);rate[6]=5.
        for _ in range(100):
            p,d=e.step(target,rate,.08)
            self.assertTrue(d['position_target_clipped'])
            self.assertLessEqual(p[6],2.+1e-8)
        self.assertAlmostEqual(p[6],2.)

    def test_rejected_input_preserves_current_dynamical_state(self):
        e=self.engine();before=(e.position.copy(),e.rate.copy(),e.acceleration.copy());bad=DEFAULT_REF.copy();bad[6]=np.nan
        with self.assertRaises(ValueError):e.step(bad)
        for a,b in zip(before,(e.position,e.rate,e.acceleration)):np.testing.assert_array_equal(a,b)

    def test_hardware_profile_and_invalid_model_limits_are_rejected(self):
        cfg=load_config();cfg['hardware_enabled']=True
        with self.assertRaises(ValueError):BoundedReferenceTrajectory(DEFAULT_REF,np.tile([-2.,2.],(29,1)),'test',cfg)
        with self.assertRaises(ValueError):BoundedReferenceTrajectory(DEFAULT_REF,np.zeros((29,2)),'test')


if __name__=='__main__':unittest.main()
