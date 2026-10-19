import unittest
import numpy as np
from motion_control.centroidal_feasibility import braking_wrench


class CentroidalTests(unittest.TestCase):
    def test_standing_force_equilibrium_and_world_translation(self):
        points=np.array([[-.1,-.1,0],[.1,-.1,0],[.1,.1,0],[-.1,.1,0]])
        a=braking_wrench([0,0,.8],points,30.,[0,0,0])
        self.assertEqual(a['status'],'feasible')
        np.testing.assert_allclose(np.sum(a['forces_world_n'],axis=0),[0,0,294.3],atol=1e-8)
        b=braking_wrench([4,-5,3.8],points+[4,-5,3],30.,[0,0,0])
        self.assertEqual(b['status'],'feasible')

    def test_off_support_braking_and_insufficient_friction_are_infeasible(self):
        self.assertEqual(braking_wrench([.5,0,.8],[[0,0,0]],30.,[-1,0,0])['status'],'infeasible')
        points=[[-2,-2,0],[2,-2,0],[2,2,0],[-2,2,0]]
        self.assertEqual(braking_wrench([0,0,.8],points,30.,[4,0,0],friction=.1)['status'],'infeasible')
        self.assertEqual(braking_wrench([0,0,.8],points,30.,[4,0,0],friction=1.)['status'],'feasible')

    def test_no_contact_is_unknown_and_invalid_input_rejected(self):
        self.assertEqual(braking_wrench([0,0,.8],[],30.,[0,0,0])['status'],'unknown')
        with self.assertRaises(ValueError):braking_wrench([0,0,.8],[],0.,[0,0,0])


if __name__=='__main__':unittest.main()
