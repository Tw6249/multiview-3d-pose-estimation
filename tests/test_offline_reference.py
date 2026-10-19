import copy
import unittest
import numpy as np
from motion_control.offline_reference import regularize,retime
from motion_control.twist import DEFAULT_REF
from motion_control.root_tracking import RootTracker


class OfflineReferenceTests(unittest.TestCase):
    def rows(self,count=30):
        result=[]
        for i in range(count):
            ref=DEFAULT_REF.copy();ref[6]+=.15*np.sin(i*.2)+.06*(-1)**i
            result.append(dict(frame_id=i,timestamp_ns=i*40_000_000,accepted=True,reference=ref.tolist(),
                               root_position_m=[i*.02,0,.8],root_heading_rad=0.))
        return result

    def test_reduces_high_frequency_motion_and_preserves_original_scoring_target(self):
        rows=self.rows();frozen=copy.deepcopy(rows);out=regularize(rows,np.tile([-2.,2.],(29,1)))
        self.assertEqual(rows,frozen)
        before=np.array([r['reference'][6] for r in rows]);after=np.array([r['reference'][6] for r in out])
        self.assertLess(np.linalg.norm(np.diff(after,n=3)),.5*np.linalg.norm(np.diff(before,n=3)))
        self.assertEqual(out[15]['source_reference'],rows[15]['reference'])
        self.assertTrue(out[15]['offline_trajectory']['uses_future_estimates'])
        self.assertAlmostEqual(out[15]['reference'][0],.5,places=8)
        self.assertAlmostEqual(out[15]['reference'][1],0.,places=8)

    def test_rejected_frames_are_never_filled_or_accepted(self):
        rows=self.rows();rows[10]=dict(frame_id=10,timestamp_ns=400_000_000,accepted=False,reason='occluded')
        out=regularize(rows,np.tile([-2.,2.],(29,1)))
        self.assertEqual(out[10],rows[10])
        self.assertEqual([r['timestamp_ns'] for r in out],[r['timestamp_ns'] for r in rows])

    def test_original_root_path_is_used_for_evaluation_even_when_tracking_path_changes(self):
        first=dict(root_position_m=[1,2,.8],root_heading_rad=0.,source_root_position_m=[1,2,.8],source_root_heading_rad=0.)
        tracker=RootTracker(first,[0,0,.8],0.)
        row=dict(first,root_position_m=[1.2,2,.8],source_root_position_m=[1.5,2,.8])
        _,diagnostic=tracker.correct(row,DEFAULT_REF,[.2,0,.8],0.)
        self.assertAlmostEqual(diagnostic['position_error_m'],0.)
        self.assertAlmostEqual(diagnostic['source_position_error_m'],.3)

    def test_slowdown_preserves_path_and_invalid_frames_and_declares_new_clock(self):
        rows=self.rows();rows[2]['reference'][:2]=[1.,.5];rows[3]['accepted']=False
        original=copy.deepcopy(rows);out=retime(rows,2.)
        self.assertEqual(rows,original)
        self.assertEqual(out[2]['timestamp_ns'],160_000_000)
        self.assertEqual(out[2]['original_exposure_timestamp_ns'],80_000_000)
        self.assertEqual(out[2]['reference_clock_domain'],'retimed_offline_reference')
        self.assertEqual(out[2]['root_position_m'],rows[2]['root_position_m'])
        self.assertEqual(out[2]['reference'][:2],[.5,.25])
        self.assertEqual(out[2]['reference'][6:],rows[2]['reference'][6:])
        self.assertFalse(out[3]['accepted'])
        with self.assertRaises(ValueError):retime(rows,.5)


if __name__=='__main__':unittest.main()
