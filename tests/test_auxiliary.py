import copy
import unittest
import numpy as np
from tests.test_motion_frontend import scene
from pose_sim.auxiliary import associate_details, attachment_checks
from pose_sim.human_motion import HumanMotionEstimator
from pose_sim.multiview_temporal import MultiViewTemporal


class AuxiliaryTests(unittest.TestCase):
    def test_independent_hand_and_foot_swap_uses_observed_anchors(self):
        _,_,_,rows=scene();row=copy.deepcopy(rows[8]);before=copy.deepcopy(row)
        for view in row['views'].values():
            xy=view['keypoints_px']
            view['auxiliary_model_anchors']={s+'_ankle':dict(keypoint_px=xy[j],score=.9)
                                           for s,j in [('left',16),('right',15)]}
            for a,b in [(17,20),(18,21),(19,22)]+list(zip(range(23,28),range(28,33))):
                xy[a],xy[b]=xy[b],xy[a]
        immutable=copy.deepcopy(row);assigned=associate_details(row)
        self.assertEqual(row,immutable)
        self.assertTrue(assigned['auxiliary_association']['corrections'])
        for change in assigned['auxiliary_association']['corrections']:
            c=change['camera'];js=range(23,33) if change['group']=='palm' else range(17,23)
            np.testing.assert_allclose(np.array(assigned['views'][c]['keypoints_px'])[list(js)],
                                       np.array(before['views'][c]['keypoints_px'])[list(js)])
        for c in row['views']:
            self.assertEqual(assigned['views'][c]['keypoints_px'][:17],before['views'][c]['keypoints_px'][:17])

    def test_no_foot_anchor_or_crossed_wrist_does_not_invent_assignment(self):
        _,_,_,rows=scene();r=copy.deepcopy(rows[0])
        for v in r['views'].values():v['keypoints_px'][10]=v['keypoints_px'][9][:]
        assigned=associate_details(r)
        self.assertEqual(assigned['auxiliary_association']['corrections'],[])

    def test_scale_aware_validity_and_unknown_attachment(self):
        _,_,xyz,rows=scene()
        for scale in (.65,1.,1.35):
            points={n:p*scale for n,p in zip(rows[0]['joint_names'],xyz[0])}
            result,bad=attachment_checks(points,{n:.005 for n in points})
            self.assertFalse(bad);self.assertTrue(all(v['status']=='supported' for v in result.values()))
        del points['left_wrist'];result,bad=attachment_checks(points,{})
        self.assertEqual(result['left_palm']['status'],'unknown');self.assertFalse(bad)

    def test_correlated_palm_detachment_expires_without_new_observation(self):
        cameras,cfg,_,rows=scene();fusion=MultiViewTemporal(dict(cameras=cameras),cfg,structured=False)
        estimator=HumanMotionEstimator();previous=None
        for i,r in enumerate(rows[:16]):
            pose,d=fusion.update(r,previous)
            if i>=8:
                for j in range(23,28):pose['joints_world_m'][j][0]+=1.
            immutable=copy.deepcopy(pose);previous=estimator.update(pose,cameras,d)
            self.assertEqual(pose,immutable)
            if i>=8:
                self.assertIsNone(previous['orientations']['left_palm'])
                self.assertNotEqual(previous['joint_states'][23]['status'],'filtered')
        self.assertIsNone(previous['joints_world_m'][23])
        self.assertEqual(previous['joint_states'][23]['last_observed_timestamp_ns'],rows[7]['timestamp_ns'])

    def test_unknown_body_anchor_disables_foot_rotation_and_support(self):
        cameras,cfg,_,rows=scene();fusion=MultiViewTemporal(dict(cameras=cameras),cfg,structured=False)
        e=HumanMotionEstimator({'ground_height_m':0.});previous=None
        for r in rows[:8]:
            pose,d=fusion.update(r,previous);pose['joints_world_m'][15]=None;pose['quality'][15]=None
            previous=e.update(pose,cameras,d)
        self.assertIsNone(previous['orientations']['left_foot'])
        self.assertEqual(previous['support']['feet']['left']['state'],'unknown')


if __name__=='__main__':unittest.main()
