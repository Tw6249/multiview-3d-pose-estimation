import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
from contextlib import redirect_stdout
import io
import json
import numpy as np
from scripts.stream_vision import ReplayDecodeBudget


class ReplayDeadlineTests(unittest.TestCase):
    def test_decode_backlog_recovery_keeps_original_exposure_times(self):
        budget=ReplayDecodeBudget(100);now=300_000_000;decoded=[];skipped=[]
        for exposure in range(0,2_000_000_000,40_000_000):
            now=max(now,exposure)
            if not budget.should_decode(exposure,now):skipped.append(exposure);continue
            now+=60_000_000;budget.record_decode(60_000_000);decoded.append((exposure,now))
        self.assertGreater(len(skipped),0);self.assertGreater(len(decoded),10)
        self.assertTrue(all(t%40_000_000==0 for t,_ in decoded))
        self.assertTrue(all(end-t<=100_000_000 for t,end in decoded[1:]))
        self.assertEqual(len(skipped)+len(decoded),50)

    def test_one_slow_decode_does_not_permanently_starve_fresh_frames(self):
        budget=ReplayDecodeBudget(100);budget.record_decode(2_000_000_000)
        self.assertTrue(budget.should_decode(3_000_000_000,3_000_000_000))
        self.assertFalse(budget.should_decode(3_000_000_000,3_060_000_000))
        with self.assertRaises(ValueError):ReplayDecodeBudget(float('nan'))
        with self.assertRaises(ValueError):budget.record_decode(-1)

    def test_summary_only_replay_preserves_counters_and_uses_configured_workers(self):
        from scripts.stream_vision import run
        class Engine:
            runtime={};warmup_seconds=0;stage_timings=[]
            def process(self,packet):
                return dict(timestamp_ns=packet.timestamp_ns,joint_names=[],joint_states=[])
            def close(self):pass
        with TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'source';src.mkdir()
            (src/'config.json').write_text(json.dumps(dict(fps=25)))
            (src/'cameras.json').write_text(json.dumps(dict(cameras={'a':{},'b':{}})))
            rows=[dict(frame_id=i,timestamp_ns=i*1_000_000,views={c:dict(timestamp_ns=i*1_000_000,image='mock.png') for c in ('a','b')}) for i in range(2)]
            (src/'frames.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            with patch('scripts.stream_vision.StreamingVision',return_value=Engine()) as engine,patch('scripts.stream_vision.np.fromfile',return_value=np.zeros(1,np.uint8)),patch('scripts.stream_vision.cv2.imdecode',return_value=np.zeros((8,8,3),np.uint8)),redirect_stdout(io.StringIO()):
                result=run(src,root/'result',save_states=False)
            self.assertEqual(engine.call_args.kwargs['detector_workers'],4)
            self.assertEqual(engine.call_args.kwargs['pose_workers'],4)
            self.assertEqual(result['source']['input_frames'],2)
            self.assertGreater(result['processed'],0);self.assertEqual(result['planner_follow_count'],0)
            self.assertAlmostEqual(result['state_rate_hz'],result['processed']/result['wall_seconds'])
            self.assertFalse((root/'result/states3d.jsonl').exists())
