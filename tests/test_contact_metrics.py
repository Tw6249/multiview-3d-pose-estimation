import unittest
from motion_control.contact_metrics import FootLoadEvents,PredictionRuns,match_landing_windows


class ContactMetricTests(unittest.TestCase):
    def test_touchdown_onset_and_confirmation_distinct_no_initial_event(self):
        e=FootLoadEvents()
        for i,f in enumerate([100,100,0,0,100,100]):e.update(i*20_000_000,dict(left=f,right=100),300.)
        self.assertEqual(e.events,[dict(side='left',timestamp_ns=80_000_000,confirmed_timestamp_ns=100_000_000)])

    def test_unknown_gap_and_force_spike_do_not_invent_touchdown(self):
        for samples in [[(0,0),(20,0),(40,None),(60,100),(80,100)],[(0,0),(20,0),(200,100),(220,100)],
                        [(0,0),(20,0),(40,100),(60,0),(80,0)]]:
            e=FootLoadEvents()
            for t,f in samples:e.update(t*1_000_000,dict(left=f,right=0),300.)
            self.assertEqual(e.events,[])

    def test_prediction_runs_do_not_count_every_update_as_step(self):
        e=PredictionRuns();events=[dict(side='left'),dict(side='left'),None,dict(side='left'),dict(side='right')]
        ids=[e.update(i*40_000_000,x) for i,x in enumerate(events)]
        self.assertEqual(ids,[0,0,None,1,2]);self.assertEqual(e.runs[0]['updates'],2)

    def test_one_to_one_matching_maximizes_cardinality_without_reuse(self):
        ws=[dict(side='left',earliest_ns=100,latest_ns=300,nominal_ns=200,received_ns=80),
            dict(side='left',earliest_ns=180,latest_ns=220,nominal_ns=200,received_ns=80)]
        ts=[dict(side='left',timestamp_ns=t) for t in (200,290)]
        r=match_landing_windows(ws,ts)
        self.assertEqual(len(r['matches']),2)
        r=match_landing_windows(ws,ts[:1]);self.assertEqual(len(r['matches']),1)

    def test_wrong_side_expired_or_before_receipt_never_match(self):
        w=dict(side='left',earliest_ns=100,latest_ns=200,nominal_ns=150,received_ns=180)
        ts=[dict(side='right',timestamp_ns=190),dict(side='left',timestamp_ns=150),dict(side='left',timestamp_ns=210)]
        self.assertEqual(match_landing_windows([w],ts)['unmatched_windows'],1)
        self.assertEqual(match_landing_windows([],ts)['touchdowns'],3)

    def test_invalid_timestamps_and_windows_rejected(self):
        e=FootLoadEvents();e.update(0,dict(left=0,right=0),300.)
        with self.assertRaises(ValueError):e.update(0,dict(left=0,right=0),300.)
        with self.assertRaises(ValueError):match_landing_windows([dict(side='left',earliest_ns=2,latest_ns=1,nominal_ns=1,received_ns=0)],[])


if __name__=='__main__':unittest.main()
