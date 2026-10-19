import copy
import unittest
import numpy as np
from tests.test_motion_frontend import scene
from pose_sim.person_selection import CrossViewPersonSelector


class PersonSelectionTests(unittest.TestCase):
    def setup_scene(self):
        cameras,cfg,xyz,rows=scene();row=copy.deepcopy(rows[0])
        for v in row['views'].values():v['num_people_detected']=1
        cid='cam00';good=copy.deepcopy(row['views'][cid]);bad=copy.deepcopy(good)
        bad['keypoints_px']=(np.array(bad['keypoints_px'])+[240.,180.]).tolist()
        row['views'][cid]=dict(timestamp_ns=row['timestamp_ns'],num_people_detected=2,keypoints_px=None,scores=None,person_candidates=[bad,good])
        return cameras,cfg,row,good,cid

    def test_other_views_recover_original_person_from_false_instance(self):
        cameras,cfg,row,good,cid=self.setup_scene();before=copy.deepcopy(row)
        out=CrossViewPersonSelector(cameras,cfg).update(row)
        self.assertEqual(row,before)
        self.assertEqual(out['views'][cid]['keypoints_px'],good['keypoints_px'])
        self.assertEqual(out['person_selection']['evidence'][0]['selected_candidate'],1)

    def test_ambiguous_candidates_remain_missing(self):
        cameras,cfg,row,good,cid=self.setup_scene();row['views'][cid]['person_candidates']=[good,copy.deepcopy(good)]
        out=CrossViewPersonSelector(cameras,cfg).update(row)
        self.assertIsNone(out['views'][cid]['keypoints_px'])

    def test_expired_or_other_selected_candidates_do_not_certify_identity(self):
        cameras,cfg,row,good,cid=self.setup_scene()
        for c in ('cam01','cam02'):row['views'][c]['timestamp_ns']-=20_000_000
        out=CrossViewPersonSelector(cameras,cfg).update(row)
        self.assertIsNone(out['views'][cid]['keypoints_px'])

    def test_single_view_and_all_ambiguous_views_do_not_guess(self):
        cameras,cfg,row,good,cid=self.setup_scene()
        for c,v in row['views'].items():
            if c!=cid:v['num_people_detected']=2
        self.assertIsNone(CrossViewPersonSelector(cameras,cfg).update(row)['views'][cid]['keypoints_px'])

    def test_existing_unique_observations_and_labels_are_unchanged(self):
        cameras,cfg,row,good,cid=self.setup_scene();row['views'][cid]=good
        for c in row['views']:
            out=CrossViewPersonSelector(cameras,cfg).update(row)
            self.assertEqual(out['views'][c],row['views'][c])


if __name__=='__main__':unittest.main()
