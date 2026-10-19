import copy
import unittest
from pose_sim.fault_injection import FAULT_TYPES,DURATIONS_MS,inject,mirror,plan
from pose_sim.landmarks import HUMAN33


def rows(n=200,cams=('0','1','2')):
    out=[]
    for f in range(n):
        views={c:dict(timestamp_ns=f*40_000_000,num_people_detected=1,
                      keypoints_px=[[float(f),float(i)] for i in range(len(HUMAN33))],scores=[1.]*len(HUMAN33),
                      auxiliary_model_anchors=dict(left_ankle=dict(keypoint_px=[1.,1.],score=5.),
                                                   right_ankle=dict(keypoint_px=[2.,2.],score=5.)))
               for c in cams}
        out.append(dict(frame_id=f,timestamp_ns=f*40_000_000,joint_names=list(HUMAN33),views=views))
    return out


def event(kind,**extra):
    return dict(id='x',type=kind,duration_ms=120,start_frame=10,end_frame=13,cameras=['1'],
                affected_joints=[],landmarks=None,params={},**extra)


class FaultInjectionTests(unittest.TestCase):
    def test_plan_is_deterministic_and_covers_every_fault_over_seeds(self):
        data=rows(750)
        self.assertEqual(plan(data,1,25),plan(data,1,25))
        seen={(e['type'],e['duration_ms']) for s in range(3) for e in plan(data,s,25)}
        self.assertEqual(seen,{(f,d) for f in FAULT_TYPES for d in DURATIONS_MS})
        self.assertTrue(all(e['end_frame']+25<=750 for e in plan(data,0,25)))

    def test_source_rows_are_not_modified(self):
        data=rows(750);before=copy.deepcopy(data)
        for s in range(3):inject(data,plan(data,s,25))
        self.assertEqual(data,before)

    def test_swap_mirrors_landmarks_and_anchors_in_target_view_only(self):
        data=rows();out,labels=inject(data,[event('view_lr_swap')])
        i,j=HUMAN33.index('left_knee'),HUMAN33.index(mirror('left_knee'))
        self.assertEqual(out[11]['views']['1']['keypoints_px'][i],data[11]['views']['1']['keypoints_px'][j])
        self.assertEqual(out[11]['views']['1']['auxiliary_model_anchors']['left_ankle']['keypoint_px'],[2.,2.])
        self.assertEqual(out[11]['views']['0'],data[11]['views']['0'])
        self.assertEqual(out[13],data[13]);self.assertEqual(labels[0]['applied_frames'],3)

    def test_time_offset_keeps_current_stamp_with_stale_content(self):
        out,_=inject(rows(),[event('view_time_offset',)|dict(params=dict(lag_frames=2))])
        view=out[12]['views']['1']
        self.assertEqual(view['timestamp_ns'],12*40_000_000);self.assertEqual(view['keypoints_px'][0],[10.,0.])

    def test_limb_dropout_suppresses_detail_landmarks_and_anchor(self):
        e=event('limb_dropout')|dict(cameras=['0','1','2'],params=dict(side='left',limb='leg'),
            landmarks=['left_knee','left_ankle','left_big_toe','left_small_toe','left_heel'])
        out,_=inject(rows(),[e])
        view=out[10]['views']['2']
        self.assertTrue(all(view['scores'][HUMAN33.index(n)]==0 for n in e['landmarks']))
        self.assertEqual(view['scores'][HUMAN33.index('right_knee')],1.)
        self.assertNotIn('left_ankle',view['auxiliary_model_anchors'])

    def test_unusable_view_is_not_counted_as_applied(self):
        data=rows()
        for f in range(10,13):data[f]['views']['1'].update(keypoints_px=None,scores=None)
        _,labels=inject(data,[event('view_dropout')])
        self.assertEqual(labels[0]['applied_frames'],0)


if __name__=='__main__':unittest.main()
