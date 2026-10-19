import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from PIL import Image
from pose_sim.pipeline import detect, reconstruct, read_jsonl

ROOT = Path(__file__).resolve().parents[1]


class RealContractTests(unittest.TestCase):
    def test_real_input_without_truth_and_blank_frame_rejection(self):
        # End-to-end contract check, using real-image importer and actual detector.
        # No ground-truth file exists anywhere in the temporary session.
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'真实 输入'
            p.mkdir()
            Image.new('RGB', (640, 640), (64, 64, 64)).save(p/'blank.png')
            camera = {'K': [[750, 0, 320], [0, 750, 320], [0, 0, 1]],
                      'R_world_to_camera': [[1,0,0],[0,1,0],[0,0,1]],
                      't_world_to_camera_m': [0,0,0], 'distortion': [0]*5,
                      'width': 640, 'height': 640}
            (p/'cameras.json').write_text(json.dumps({'cameras': {'a': camera, 'b': camera}}))
            with (p/'frames.csv').open('w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['camera_id', 'timestamp_ns', 'image'])
                w.writerows([['a', 0, 'blank.png'], ['a', 33_333_333, 'blank.png'],
                             ['b', 1_000_000, 'blank.png'], ['b', 60_000_000, 'blank.png']])
            out = p/'session'
            subprocess.run([sys.executable, str(ROOT/'scripts/import_real.py'), '--csv', str(p/'frames.csv'),
                            '--calibration', str(p/'cameras.json'), '--output', str(out)], check=True, capture_output=True)
            frames = list(read_jsonl(out/'frames.jsonl'))
            self.assertEqual(len(frames[0]['views']), 2)
            self.assertEqual(len(frames[1]['views']), 1)
            self.assertFalse((out/'ground_truth.jsonl').exists())
            detect(out)
            reconstruct(out)
            detections = list(read_jsonl(out/'detections.jsonl'))
            self.assertTrue(all(v['num_people_detected'] == 0 for r in detections for v in r['views'].values()))
            self.assertTrue(all(pose is None for r in read_jsonl(out/'poses3d.jsonl') for pose in r['joints_world_m']))


if __name__ == '__main__':
    unittest.main()
