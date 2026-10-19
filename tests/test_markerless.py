"""Model adapter contract tests; these do not measure robot recognition accuracy."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
from pose_sim.markerless import OnnxRobotDetector,decode_landmarks


class MarkerlessAdapterTests(unittest.TestCase):
    def test_non_square_resize_pixel_centres_and_invalid_locations(self):
        points,scores,instances = decode_landmarks(np.array([[[[99.5,49.5,0.9],[-10,20,1],[300,100,0]]]]),
                                                   3,(800,600),(400,200))
        np.testing.assert_allclose(points[0],[199.5,149.5])
        self.assertEqual(points[1:],[None,None]); self.assertEqual(scores[1:],[0,0])
        self.assertEqual(instances,1)

    def test_no_or_multiple_instances_are_not_matched_across_cameras(self):
        for count in (0,2):
            points,scores,instances = decode_landmarks(np.zeros((1,count,4,3)),4,(640,640),(320,320))
            self.assertIsNone(points); self.assertIsNone(scores); self.assertEqual(instances,count)

    def test_bad_tensor_score_or_nan_rejected(self):
        invalid = [np.zeros((1,4,3)), np.full((1,1,4,3),np.nan), np.full((1,1,4,3),2.)]
        for value in invalid:
            with self.assertRaises(ValueError): decode_landmarks(value,4,(640,640),(320,320))

    def make_runtime(self):
        source = SimpleNamespace(sha256='geometry_hash',names=['base/a','base/b','base/c'])
        ort = Mock()
        ort.get_available_providers.return_value = ['CUDAExecutionProvider','CPUExecutionProvider']
        session = ort.InferenceSession.return_value
        session.get_providers.return_value = ['CPUExecutionProvider']
        session.get_modelmeta.return_value.custom_metadata_map = {
            'perception_mode':'markerless','source_model_sha256':source.sha256,
            'input_encoding':'rgb_float32_0_1_stretch','output_encoding':'image_xy_score',
            'landmark_names':json.dumps(source.names)}
        session.get_inputs.return_value = [SimpleNamespace(type='tensor(float)',shape=[1,3,4,4],name='rgb')]
        session.get_outputs.return_value = [SimpleNamespace(type='tensor(float)',name='landmarks')]
        return source,ort,session

    def test_image_channel_encoding_and_model_metadata_binding(self):
        source,ort,session = self.make_runtime()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'model.onnx'; path.write_bytes(b'fake graph used with mocked runtime only')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.dict('sys.modules',{'onnxruntime':ort}):
                detector = OnnxRobotDetector(source,path,digest,'cpu')
                session.run.return_value = [np.array([[[[1,1,0.9],[2,1,0.8],[1,2,0.7]]]],np.float32)]
                image = np.zeros((4,4,3),np.uint8); image[:] = [20,40,80]
                detector.detect_image(image)
                feed = session.run.call_args[0][1]['rgb']
                np.testing.assert_allclose(feed[0,:,0,0],np.array([80,40,20])/255,rtol=1e-6)
                self.assertEqual(feed.dtype,np.float32)
                session.disable_fallback.assert_called_once()
                session.get_modelmeta.return_value.custom_metadata_map['source_model_sha256'] = 'wrong'
                with self.assertRaisesRegex(ValueError,'metadata'):
                    OnnxRobotDetector(source,path,digest,'cpu')
                with self.assertRaisesRegex(ValueError,'checksum'):
                    OnnxRobotDetector(source,path,'wrong','cpu')

    def test_requested_cuda_cannot_silently_fall_back_to_cpu(self):
        source,ort,session = self.make_runtime()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'model.onnx'; path.write_bytes(b'fake graph')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.dict('sys.modules',{'onnxruntime':ort}),patch('pose_sim.markerless.preload_cuda'):
                with self.assertRaisesRegex(RuntimeError,'unexpected provider'):
                    OnnxRobotDetector(source,path,digest,'cuda')


if __name__ == '__main__': unittest.main()
