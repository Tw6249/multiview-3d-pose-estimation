import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from scipy.io import savemat
from pose_sim.mpi3dhp import parse_calibration, reference_rows, BODY_MAP, WORLD_ROTATION
from pose_sim.public_download import HTTPRanges
from scripts.evaluate_mpi3dhp import packets, failure_audit


def calibration_text():
    blocks = ['Skeletool Camera Calibration File V1.0']
    for cid, tx in [(0, 0), (2, 1000)]:
        blocks.append(f'''name {cid}
  size 2048 2048
  animated 0
  intrinsic 1000 0 1024 0 0 1000 1024 0 0 0 1 0 0 0 0 1
  extrinsic 1 0 0 {tx} 0 1 0 0 0 0 1 3000 0 0 0 1
  radial 0
''')
    return '\n'.join(blocks)


class MPIImportTests(unittest.TestCase):
    def test_resize_projection_mm_and_right_handed_axes(self):
        converted, original = parse_calibration(calibration_text(), [0, 2])
        world_mm = np.array([100., 1600, 200])
        world_m = WORLD_ROTATION@world_mm/1000
        self.assertTrue(np.allclose(world_m, [.1, -.2, 1.6]))
        self.assertAlmostEqual(np.linalg.det(WORLD_ROTATION), 1)
        for cid, c in original.items():
            old = c['K']@(c['R']@world_mm+c['t']); old = old[:2]/old[2]
            new = converted['cameras'][str(cid)]
            xyz = np.array(new['R_world_to_camera'])@world_m+new['t_world_to_camera_m']
            uv = np.array(new['K'])@xyz; uv = uv[:2]/uv[2]
            self.assertTrue(np.allclose(uv, (old+.5)/2-.5))

    def test_bad_camera_models_fail_closed(self):
        for old, new in [('radial 0', 'radial 0.1'), ('animated 0', 'animated 1'),
                         ('extrinsic 1 0 0', 'extrinsic -1 0 0')]:
            with self.assertRaises(ValueError):
                parse_calibration(calibration_text().replace(old, new), [0, 2])
        with self.assertRaises(ValueError):
            parse_calibration(calibration_text(), [0, 7])

    def test_reference_mapping_excludes_normalized_lengths_and_faces(self):
        _, original = parse_calibration(calibration_text(), [0, 2])
        world = np.arange(3*28*3).reshape(3, 28, 3).astype(float)
        xyz = np.empty((2, 1), object); uv = np.empty((2, 1), object)
        for index, camera in enumerate(original.values()):
            points = world@camera['R'].T+camera['t']
            projected = points@camera['K'].T
            xyz[index, 0] = points.reshape(3, 84)
            uv[index, 0] = (projected[..., :2]/projected[..., 2:]).reshape(3, 56)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'annot.mat'
            data = dict(cameras=[[0, 2]], frames=np.arange(3)[:, None], annot3=xyz, annot2=uv,
                        univ_annot3=np.zeros((2, 1)))
            savemat(path, data)
            rows, checks = reference_rows(path, original, 1, 2)
            self.assertEqual(rows[0]['timestamp_ns'], 40_000_000)
            self.assertEqual(rows[0]['source_frame_index'], 1)
            self.assertEqual(rows[0]['joint_names'], list(BODY_MAP))
            expected = world[1, list(BODY_MAP.values())]@WORLD_ROTATION.T/1000
            self.assertTrue(np.allclose(rows[0]['joints_world_m'], expected))
            self.assertLess(checks['camera_reference_max_disagreement_mm'], 1e-6)
            with self.assertRaises(ValueError): reference_rows(path, original, 2, 2)
            rows, _ = reference_rows(path, original, 0, 2, stride=2)
            self.assertEqual([r['source_frame_index'] for r in rows], [0, 2])
            self.assertEqual(rows[1]['timestamp_ns'], 40_000_000)
            self.assertTrue(np.allclose(rows[1]['joints_world_m'], world[2, list(BODY_MAP.values())]@WORLD_ROTATION.T/1000))
            with self.assertRaises(ValueError): reference_rows(path, original, 1, 2, stride=2)
            data['frames'] += 1
            savemat(path, data)
            with self.assertRaises(ValueError): reference_rows(path, original, 0, 2)

    def test_video_replay_without_any_reference_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            video = folder/'source.avi'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 25, (32, 32))
            self.assertTrue(writer.isOpened())
            for i in range(6): writer.write(np.full((32, 32, 3), i*30, np.uint8))
            writer.release()
            cal = dict(cameras={c: dict(width=32, height=32) for c in ('0', '2')})
            source = dict(start_frame=2, frames=3, fps=25, videos={c: dict(path=str(video)) for c in ('0', '2')})
            (folder/'cameras.json').write_text(json.dumps(cal))
            (folder/'source.json').write_text(json.dumps(source))
            rows = list(packets(folder))
            self.assertEqual([r.timestamp_ns for r in rows], [80_000_000, 120_000_000, 160_000_000])
            self.assertTrue(np.allclose(rows[0].frames['0'].image_bgr.mean(), 60, atol=2))
            source['frames'] = 5
            (folder/'source.json').write_text(json.dumps(source))
            with self.assertRaises(ValueError): list(packets(folder))
            # A 50 Hz source decimated by two keeps source-frame times on a 40 ms grid.
            source.update(start_frame=1, frames=4, fps=25, frame_stride=2)
            (folder/'source.json').write_text(json.dumps(source))
            with self.assertRaises(ValueError): list(packets(folder))
            source['frames'] = 3
            (folder/'source.json').write_text(json.dumps(source))
            rows = list(packets(folder))
            self.assertEqual([r.timestamp_ns for r in rows], [20_000_000, 60_000_000, 100_000_000])
            self.assertTrue(np.allclose(rows[1].frames['2'].image_bgr.mean(), 90, atol=2))


class RangeContractTests(unittest.TestCase):
    def test_ignored_or_truncated_range_is_rejected(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, n): return b'abc'
        source = HTTPRanges.__new__(HTTPRanges)
        source.url = 'https://example.invalid/data'; source.size = 10; source.etag = None
        for status, header in [(200, 'bytes 0-3/10'), (206, 'bytes 1-4/10'), (206, 'bytes 0-3/10')]:
            response = Response(); response.status = status; response.headers = {'Content-Range': header}
            with patch('urllib.request.urlopen', return_value=response), patch('time.sleep'):
                with self.assertRaises(ValueError): source.range(0, 3)


class PublicFailureMetricsTests(unittest.TestCase):
    def test_bad_predictions_are_counted_and_good_raw_recovery_is_visible(self):
        reference = dict(frame_id=0, source_frame_index=99, joint_names=['right_wrist'], joints_world_m=[[0., 0, 0]])
        raw = dict(joint_names=['right_wrist'], joints_world_m=[[.1, 0, 0]], quality=[{}])
        current = dict(joint_names=['right_wrist'], joints_world_m=[[1.2, 0, 0]],
                       joint_states=[dict(status='predicted', reason='temporal innovation outlier')])
        result = failure_audit([current], [raw], [reference])
        self.assertEqual(result['outputs_over_500mm'], 1)
        self.assertEqual(result['current_observations_over_500mm'], 0)
        self.assertEqual(result['current_over_500mm_while_raw_under_150mm'], 1)
        self.assertEqual(result['worst_examples'][0]['source_frame_index'], 99)


if __name__ == '__main__': unittest.main()
