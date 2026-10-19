from types import SimpleNamespace
import unittest
import numpy as np
from pose_sim.inference import suppress_boxes, CpuNmsDetector, pose_batch,select_landmarks,infer_person_candidates,complete_selected_candidates


class VisionInferenceTests(unittest.TestCase):
    def test_out_of_image_predictions_are_not_visible_measurements(self):
        points=np.array([[5,6],[-1,6],[640,5],[5,480]])
        k,s,bad=select_landmarks(points,np.ones(4),[0,1,2,3],640,480)
        np.testing.assert_array_equal(k,points)
        np.testing.assert_array_equal(s,[1,0,0,0]);self.assertEqual(bad,[1,2,3])

    def test_nms_keeps_separate_people_and_removes_duplicate_and_low_scores(self):
        boxes=np.array([[0,0,20,20],[1,1,21,21],[80,0,100,20],[30,0,40,20]],np.float32)
        result=suppress_boxes(boxes,np.array([.9,.8,.7,.2]))
        np.testing.assert_allclose(result,boxes[[0,2]])
        self.assertEqual(suppress_boxes(boxes,np.zeros(4)).shape,(0,4))

    def test_nonperson_objects_do_not_count_as_additional_people(self):
        boxes=np.array([[[0,0,20,20],[80,0,100,20]]],np.float32)
        scores=np.zeros((1,80,2),np.float32);scores[0,0,0]=.9;scores[0,56,1]=.95
        model=SimpleNamespace(preprocess=lambda image:(image,2.),inference=lambda image:(boxes,scores))
        det=CpuNmsDetector(model,dict(score_threshold=.3,iou_threshold=.65,max_output=200))
        np.testing.assert_allclose(det(np.zeros((2,2,3))),[[0,0,10,10]])

    def test_batch_applies_each_camera_crop_transform(self):
        class Session:
            def get_inputs(self):return [SimpleNamespace(name='input',shape=['batch',3,2,2])]
            def run(self,names,feed):
                n=feed['input'].shape[0]
                return [np.ones((n,1,2)),np.ones((n,1))]
        class Model:
            session=Session()
            def preprocess(self,image,box):return image,np.array(box[:2]),np.array([2,2])
            def postprocess(self,outputs,center,scale):return outputs[0]*scale+center,outputs[1]
        images=[np.zeros((2,2,3))]*2
        results=pose_batch(Model(),images,[[10,20,30,40],[100,200,300,400]])
        np.testing.assert_allclose(results[0][0],[[12,22]])
        np.testing.assert_allclose(results[1][0],[[102,202]])
        self.assertEqual(pose_batch(Model(),[],[]),[])

    def test_candidate_selection_defers_details_and_completes_only_selected_person(self):
        calls={'body':0,'details':0}
        def body_model(image,bboxes):
            calls['body']+=1
            return np.full((1,17,2),bboxes[0][0]+10.,dtype=np.float32),np.ones((1,17),np.float32)
        def detail_model(image,bboxes):
            calls['details']+=1
            self.assertEqual(np.asarray(bboxes[0]).dtype,np.dtype('float32'))
            return np.full((1,133,2),50.,dtype=np.float32),np.full((1,133),.8,np.float32)
        body=SimpleNamespace(body_pose_model=body_model,pose_model=detail_model)
        camera=dict(width=640,height=480);indices=list(range(17))+[17,18]
        candidates=infer_person_candidates(body,'hybrid',indices,camera,None,[[1,2,30,40],[80,2,100,40]],body_only=True)
        self.assertEqual(calls,{'body':2,'details':0})
        self.assertEqual(candidates[0]['scores'][17:],[0,0])
        selected=dict(candidates[1]);row=dict(views={'cam':selected},person_selection=dict(evidence=[dict(camera='cam',selected_candidate=1)]))
        complete_selected_candidates(body,'hybrid',indices,{'cam':camera},{'cam':None},row)
        self.assertEqual(calls,{'body':2,'details':1})
        self.assertEqual(row['views']['cam']['keypoints_px'][:17],candidates[1]['keypoints_px'][:17])
        np.testing.assert_allclose(row['views']['cam']['scores'][17:],[.8,.8])
        self.assertFalse(row['views']['cam']['candidate_body_only'])


if __name__=='__main__':unittest.main()
