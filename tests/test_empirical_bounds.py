import unittest
import numpy as np
from pose_sim.empirical_bounds import fit_bounds,radius,block_maxima


class EmpiricalBoundsTests(unittest.TestCase):
    def test_domain_fingerprint_and_finite_sample_rank(self):
        c=fit_bounds(list(range(1,21)),'a','hash',.9)
        self.assertEqual(c['rank'],19);self.assertEqual(radius(c,.01,'a','hash'),.19)
        self.assertIsNone(radius(c,.01,'other','hash'));self.assertIsNone(radius(c,.01,'a','changed'))
        self.assertFalse(c['guarantee_claimed'])

    def test_blocks_do_not_count_correlated_joints_as_independent_samples(self):
        t=np.repeat(np.arange(25,dtype=np.int64)*40_000_000,12)
        a=block_maxima(np.ones(300),np.ones(300),t,np.ones(300,bool),.005)
        self.assertEqual(a,[1.])

    def test_persisted_normalization_floor_is_used_for_small_reported_std(self):
        c=fit_bounds([2.]*20,'a','hash',std_floor=.005)
        self.assertEqual(radius(c,.001,'a','hash'),.01)
        self.assertEqual(radius(c,.02,'a','hash'),.04)
        with self.assertRaises(ValueError):fit_bounds([2.]*20,'a','hash',std_floor=float('nan'))
        with self.assertRaises(ValueError):block_maxima([1.],[1.],np.array([0]),[True],float('nan'))

    def test_missing_and_predicted_cannot_improve_calibration_coverage(self):
        a=block_maxima([.1,100,np.nan],[.01,.01,.01],np.array([0,0,0],dtype=np.int64),[True,False,True],.005)
        self.assertEqual(a,[10.])
        with self.assertRaises(ValueError):fit_bounds([1]*19,'a','b')
        with self.assertRaises(ValueError):fit_bounds([float('nan')]*20,'a','b')


if __name__=='__main__':unittest.main()
