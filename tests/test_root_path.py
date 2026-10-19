import unittest
import numpy as np
from motion_control.root_path import ScaledRootPath


class RootPathTests(unittest.TestCase):
    def test_size_updates_preserve_translation_invariance_and_velocity(self):
        a,b=ScaledRootPath(),ScaledRootPath()
        previous=np.zeros(2)
        for i,scale in enumerate((.6,.7,.9,.8)):
            source=np.array([i*.04,-i*.02])
            path=a.update(source,scale,i*40_000_000)
            shifted=b.update(source+[20.,-15.],scale,i*40_000_000)
            np.testing.assert_allclose(path,shifted,atol=1e-14)
            if i:np.testing.assert_allclose((path-previous)/.04,scale*np.array([1.,-.5]),atol=1e-14)
            previous=path

    def test_stationary_pelvis_does_not_move_when_size_estimate_changes(self):
        path=ScaledRootPath()
        for i,scale in enumerate((.5,.9,1.2)):
            np.testing.assert_array_equal(path.update([20.,-15.],scale,i),[0,0])

    def test_rejected_input_and_returned_arrays_cannot_change_history(self):
        path=ScaledRootPath();result=path.update([1.,2.],.5,10)
        result[:]=100
        for point,scale,time in (([2.,3.],1.,10),([np.nan,0],1.,20),([2,3],0.,20)):
            with self.assertRaises(ValueError):path.update(point,scale,time)
        np.testing.assert_allclose(path.update([1.2,2.4],.5,20),[.1,.2],atol=1e-14)


if __name__=='__main__':unittest.main()
