import unittest
import numpy as np

from pose_sim.sensors import ExposureFrame
from pose_sim.spatial_evidence import SimCCEvidence, capture_simcc


def example(**changes):
    args = dict(camera_id='left', exposure_timestamp_ns=1_800_000_000_000_000_007,
        clock_domain='hardware', model_sha256='a'*64, joint_names=('wrist',),
        image_size_wh=(640,480), input_size_wh=(4,6), center_px=np.array([150.,210.]),
        scale_px=np.array([80.,120.]), x_response=np.arange(8,dtype=np.float32)[None]-4,
        y_response=np.arange(12,dtype=np.float32)[None]-6)
    return SimCCEvidence(**(args | changes))


class SpatialEvidenceTests(unittest.TestCase):
    def test_crop_coordinates_and_outside_support_remain_explicit(self):
        e=example();bins=np.array([[0.,0.],[4.,6.],[8.,12.],[-1.,20.]])
        np.testing.assert_allclose(e.bins_to_pixels(bins)[:3],[[110,150],[150,210],[190,270]])
        np.testing.assert_allclose(e.pixels_to_bins(e.bins_to_pixels(bins)),bins)

    def test_responses_are_owned_and_not_claimed_as_probability(self):
        raw=np.arange(8,dtype=np.float32)[None]-4;e=example(x_response=raw);raw[:]=100
        self.assertEqual(e.x_response[0,0],-4)
        with self.assertRaises(ValueError):e.x_response[0,0]=0
        self.assertFalse(e.semantics['calibrated_probability'])
        self.assertFalse(e.semantics['joint_xy_distribution'])
        self.assertIsNone(e.semantics['visibility_probability'])
        self.assertEqual(e.response_bytes,80)
        self.assertEqual(e.exposure_timestamp_ns,1_800_000_000_000_000_007)

    def test_invalid_layout_identity_and_geometry_rejected(self):
        for change in [dict(exposure_timestamp_ns=1.5),dict(camera_id=''),dict(model_sha256='bad'),
            dict(joint_names=('wrist','wrist')),dict(input_size_wh=(0,6)),
            dict(scale_px=np.array([0.,1.])),dict(split_ratio=float('nan')),
            dict(x_response=np.ones((1,9))),dict(y_response=np.full((1,12),np.nan))]:
            with self.subTest(change=change),self.assertRaises(ValueError):example(**change)

    def test_adapter_single_inference_keeps_exposure_and_decoder(self):
        class Model:
            model_input_size=(4,6);to_openpose=False;calls=0
            def preprocess(self,image,box):return image,np.array([150.,210.]),np.array([80.,120.])
            def inference(self,image):
                self.calls+=1
                return [np.ones((1,1,8),np.float32),np.ones((1,1,12),np.float32)]
            def postprocess(self,outputs,center,scale):return np.array([[[9.,8.]]]),np.array([[.7]])
        frame=ExposureFrame('c',11,20,'device',np.zeros((48,64,3),np.uint8));model=Model()
        k,s,e=capture_simcc(model,frame,[1,2,30,40],['wrist'],'b'*64)
        self.assertEqual(model.calls,1);np.testing.assert_array_equal(k,[[9,8]])
        self.assertEqual(e.exposure_timestamp_ns,11);self.assertEqual(e.clock_domain,'device')
        self.assertEqual(e.image_size_wh,(64,48));self.assertEqual(s[0],.7)
        for box in [None,[1,2,1,20],[float('nan'),0,10,20]]:
            with self.assertRaises(ValueError):capture_simcc(model,frame,box,['wrist'],'b'*64)
        self.assertEqual(model.calls,1)


if __name__=='__main__':unittest.main()
