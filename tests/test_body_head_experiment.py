import copy
import unittest
import numpy as np
from pose_sim.body_head_experiment import replace_body
from pose_sim.landmarks import HUMAN33


class BodyHeadTests(unittest.TestCase):
    def test_only_body_pixels_change_no_temporal_or_identity_change(self):
        row=dict(joint_names=HUMAN33,frame_id=3,timestamp_ns=123,perception_signature='old',views={'a':dict(timestamp_ns=125,
            person_bbox_xyxy_px=[0,0,90,90],keypoints_px=[[1.,2.]]*33,scores=[.9]*33,auxiliary_model_anchors={'old':True})})
        original=copy.deepcopy(row);k=np.ones((17,2))*5;k[0,0]=101
        out=replace_body(row,{'a':(k,np.ones(17))},{'a':dict(width=100,height=100)},'new')
        self.assertEqual(row,original);self.assertEqual(out['timestamp_ns'],123)
        self.assertEqual(out['views']['a']['timestamp_ns'],125)
        self.assertEqual(out['views']['a']['keypoints_px'][17:],row['views']['a']['keypoints_px'][17:])
        self.assertEqual(out['views']['a']['auxiliary_model_anchors'],{'old':True})
        self.assertEqual(out['views']['a']['scores'][0],0)
        self.assertNotEqual(out['perception_signature'],'old')

    def test_missing_crop_and_nonfinite_body_rejected(self):
        row=dict(joint_names=HUMAN33,perception_signature='x',views={'a':dict(keypoints_px=None)})
        with self.assertRaises(ValueError):replace_body(row,{'a':(np.ones((17,2)),np.ones(17))},{},'x')
        row['views']['a']=dict(keypoints_px=[[1,2]]*33,scores=[1]*33,person_bbox_xyxy_px=[0,0,90,90])
        for k,s in [(np.ones((16,2)),np.ones(17)),(np.full((17,2),np.nan),np.ones(17))]:
            with self.assertRaises(ValueError):replace_body(row,{'a':(k,s)},{},'x')
