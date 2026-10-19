import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.import_real import nearest_unused_index
from pose_sim.pipeline import read_jsonl

ROOT = Path(__file__).resolve().parents[1]


class TimestampMatchingTests(unittest.TestCase):
    def test_used_nearest_falls_back_to_unused_frame_in_imported_packets(self):
        # Previously the second anchor picked b/4ms again and lost b/12ms.
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p/'image.png').write_bytes(b'importer copies image bytes without inference')
            camera = {'K': [[750,0,320],[0,750,320],[0,0,1]],
                      'R_world_to_camera': [[1,0,0],[0,1,0],[0,0,1]],
                      't_world_to_camera_m': [0,0,0], 'width': 640, 'height': 640}
            (p/'cameras.json').write_text(json.dumps({'cameras': {'a': camera, 'b': camera}}))
            with (p/'frames.csv').open('w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['camera_id', 'timestamp_ns', 'image'])
                w.writerows([(cid, ms*1_000_000, 'image.png') for cid, ms in
                             [('a', 0), ('a', 8), ('a', 9), ('a', 30), ('b', 4), ('b', 12)]])
            subprocess.run([sys.executable, str(ROOT/'scripts/import_real.py'),
                '--csv', str(p/'frames.csv'), '--calibration', str(p/'cameras.json'),
                '--output', str(p/'out')], check=True, capture_output=True)
            rows = list(read_jsonl(p/'out/frames.jsonl'))
            self.assertEqual([r['views'].get('b', {}).get('timestamp_ns') for r in rows],
                             [4_000_000, 12_000_000, None, None])
            self.assertEqual([len(r['views']) for r in rows], [2, 2, 1, 1])
            self.assertEqual((p/'out'/rows[1]['views']['b']['image']).read_bytes(),
                             (p/'image.png').read_bytes())

    def test_tolerance_boundary_and_deterministic_ties(self):
        self.assertEqual(nearest_unused_index([4, 12], 8, 4, set()), 0)
        self.assertEqual(nearest_unused_index([4, 12], 8, 4, {0}), 1)
        self.assertIsNone(nearest_unused_index([4, 12], 8, 3, set()))
        self.assertIsNone(nearest_unused_index([4, 12], 8, 4, {0, 1}))
        self.assertIsNone(nearest_unused_index([], 8, 4, set()))

    def test_nanosecond_timestamps_keep_integer_precision(self):
        base = 1_800_000_000_000_000_000
        self.assertEqual(nearest_unused_index([base, base+2], base+1, 1, {0}), 1)
        self.assertIsNone(nearest_unused_index([base], base+1, 0, set()))


if __name__ == '__main__':
    unittest.main()
