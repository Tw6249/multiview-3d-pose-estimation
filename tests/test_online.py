import time
import unittest
import numpy as np
from pose_sim.online import ExposurePacket,ExposureSynchronizer,LatestMailbox,consume_latest
from pose_sim.sensors import ExposureFrame


def frame(c,t,arrival=None):
    return ExposureFrame(c,t,t if arrival is None else arrival,'host_monotonic',np.zeros((8,8,3),np.uint8))


class OnlineTests(unittest.TestCase):
    def test_bounded_mailbox_keeps_latest_without_timestamp_reuse(self):
        q=LatestMailbox()
        for t in range(1,20):q.submit(ExposurePacket(t,t,{c:frame(c,t) for c in ('a','b')}))
        self.assertEqual(q.replaced,18);self.assertEqual(q.get().timestamp_ns,19)
        with self.assertRaises(ValueError):q.submit(ExposurePacket(20,19,{c:frame(c,19) for c in ('a','b')}))
        q.close();self.assertIsNone(q.get())

    def test_sync_drops_stale_and_never_reuses_exposures(self):
        sync=ExposureSynchronizer(['a','b'],max_skew_ns=5,max_age_ns=30)
        self.assertIsNone(sync.push(frame('a',10),50))
        self.assertIsNone(sync.push(frame('a',50),50))
        p=sync.push(frame('b',53),53);self.assertEqual(p.timestamp_ns,50)
        self.assertIsNone(sync.push(frame('b',60),60))
        with self.assertRaises(ValueError):sync.push(frame('a',50),60)

    def test_skew_is_pairwise_not_only_distance_to_anchor(self):
        with self.assertRaises(ValueError):
            ExposurePacket(0,0,{'a':frame('a',0),'b':frame('b',8_000_000)}).validate()

    def test_slow_processing_marks_result_stale(self):
        q=LatestMailbox();t=time.perf_counter_ns();q.submit(ExposurePacket(0,t,{c:frame(c,t) for c in ('a','b')}));q.close()
        def slow(packet):time.sleep(.015);return {}
        results=[];stats=consume_latest(q,slow,results.append,max_output_age_ms=1)
        self.assertEqual(stats['stale_after'],1);self.assertFalse(results[0]['delivery']['valid_for_new_motion'])

    def test_long_running_metrics_are_bounded_without_losing_total_counters(self):
        class Source:
            closed=False;submitted=10;replaced=0;count=0
            def get(self):
                if self.count==10:self.closed=True;return None
                self.count+=1;t=time.perf_counter_ns()
                return ExposurePacket(self.count,t,{c:frame(c,t) for c in ('a','b')})
        stats=consume_latest(Source(),lambda packet:{},lambda row:None,metrics_capacity=3)
        self.assertEqual(stats['processed'],10);self.assertEqual(len(stats['latency_ms']),3)
        self.assertEqual(len(stats['processing_ms']),3);self.assertTrue(stats['metrics_window']['truncated'])


if __name__=='__main__':unittest.main()
