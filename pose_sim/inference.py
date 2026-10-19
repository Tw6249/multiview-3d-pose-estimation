"""Batch independent camera crops without modifying the exported ONNX graph."""
import numpy as np
import cv2
import time


def select_landmarks(keypoints,scores,indices,width,height):
    k=np.asarray(keypoints)[indices].copy();s=np.asarray(scores)[indices].copy()
    outside=(k[:,0]<0)|(k[:,0]>=width)|(k[:,1]<0)|(k[:,1]>=height)
    s[outside]=0
    return k,s,np.flatnonzero(outside).tolist()


def infer_person_candidate(body,model_kind,indices,camera,image,box,body_only=False,body_candidate=None):
    """Infer one crop; hybrid selection may defer unneeded hand/foot inference."""
    anchors={}
    if model_kind=='hybrid' and body_only:
        bk,bs=body.body_pose_model(image,bboxes=[box])
        k=np.zeros((133,2),dtype=bk.dtype);s=np.zeros(133,dtype=bs.dtype)
        k[:17]=bk[0];s[:17]=bs[0]
    else:
        k,s=body.pose_model(image,bboxes=[box]);k,s=k[0],s[0];anchors={}
        if model_kind=='hybrid':
            anchors={n:dict(keypoint_px=k[j].tolist(),score=float(s[j]))
                for n,j in (('left_ankle',15),('right_ankle',16)) if np.isfinite(k[j]).all()
                and np.isfinite(s[j]) and 0<=k[j][0]<camera['width'] and 0<=k[j][1]<camera['height']}
            if body_candidate is None:
                bk,bs=body.body_pose_model(image,bboxes=[box]);k[:17]=bk[0];s[:17]=bs[0]
            else:
                for j,index in enumerate(indices):
                    if index<17:
                        k[index]=body_candidate['keypoints_px'][j];s[index]=body_candidate['scores'][j]
    k,s,outside=select_landmarks(k,s,indices,camera['width'],camera['height'])
    if not np.isfinite(k).all() or not np.isfinite(s).all():return None
    return dict(person_bbox_xyxy_px=np.asarray(box[:4],float).tolist(),keypoints_px=k.tolist(),scores=s.tolist(),
                outside_image_landmarks=outside,auxiliary_model_anchors=anchors,
                candidate_body_only=bool(body_only and model_kind=='hybrid'))


def infer_person_candidates(body,model_kind,indices,camera,image,boxes,body_only=False):
    """Fresh RGB inference for two to four detector instances; no identity choice."""
    if not 2<=len(boxes)<=4:return []
    return [candidate for box in boxes
            if (candidate:=infer_person_candidate(body,model_kind,indices,camera,image,box,body_only)) is not None]


def complete_selected_candidates(body,model_kind,indices,cameras,images,row):
    """Complete only already-selected crops in an owned row; no identity change.

    Body-only candidates have zero confidence for all uncomputed details. If the
    full inference fails finite-value checks, those details remain unobserved.
    The selected body landmarks are retained exactly; details cannot alter choice.
    """
    for evidence in row.get('person_selection',{}).get('evidence',[]):
        if evidence['selected_candidate'] is None:continue
        cid=evidence['camera'];view=row['views'][cid]
        if not view.get('candidate_body_only',False):continue
        begin=time.perf_counter()
        # Preserve the detector's float32 crop arithmetic after serializing its
        # box to lists; float64 promotion changes affine-warp pixels and scores.
        box=np.asarray(view['person_bbox_xyxy_px'],dtype=np.float32)
        candidate=infer_person_candidate(body,model_kind,indices,cameras[cid],images[cid],box,body_candidate=view)
        view['inference_seconds']=view.get('inference_seconds',0.)+time.perf_counter()-begin
        if candidate is not None:
            for j,index in enumerate(indices):
                if index<17:
                    candidate['keypoints_px'][j]=view['keypoints_px'][j]
                    candidate['scores'][j]=view['scores'][j]
            view.update(candidate)
    return row


def suppress_boxes(boxes, scores, score_threshold=.3, iou_threshold=.65, max_output=200):
    boxes,scores=np.asarray(boxes),np.asarray(scores)
    valid=(scores>score_threshold)&np.isfinite(scores)&np.isfinite(boxes).all(axis=1)
    boxes,scores=boxes[valid],scores[valid]
    if not len(boxes):return np.empty((0,4),dtype=np.float32)
    xywh=boxes.copy();xywh[:,2:]-=xywh[:,:2]
    ids=np.asarray(cv2.dnn.NMSBoxes(xywh.tolist(),scores.tolist(),score_threshold,iou_threshold),dtype=int).ravel()
    return boxes[ids[:max_output]].copy()


class CpuNmsDetector:
    def __init__(self, model, config):
        self.model,self.config=model,config

    def __getattr__(self,name):
        return getattr(self.model,name)

    def __call__(self,image):
        tensor,ratio=self.model.preprocess(image)
        boxes,scores=self.model.inference(tensor)
        if boxes.ndim!=3 or scores.ndim!=3 or scores.shape[0]!=1 or scores.shape[1]<1:
            raise ValueError('CPU NMS adapter requires single-image YOLOX outputs')
        # The pinned HumanArt YOLOX keeps COCO's 80 classes. Class 0 is person;
        # chairs/objects must not be counted as additional people.
        return suppress_boxes(boxes[0],scores[0,0],self.config['score_threshold'],
                              self.config['iou_threshold'],self.config['max_output'])/ratio


def pose_batch(model, images, boxes):
    if not images:
        return []
    prepared = [model.preprocess(image, box) for image, box in zip(images, boxes)]
    shape = model.session.get_inputs()[0].shape
    if isinstance(shape[0], int) and shape[0] != len(prepared):
        # Models exported with a fixed batch retain their supported path.
        return [(k[0], s[0]) for k, s in (model(im, bboxes=[box]) for im, box in zip(images, boxes))]
    tensor = np.ascontiguousarray(np.stack([p[0].transpose(2, 0, 1) for p in prepared]), dtype=np.float32)
    outputs = model.session.run(None, {model.session.get_inputs()[0].name: tensor})
    result = []
    for i, (_, center, scale) in enumerate(prepared):
        k, s = model.postprocess([out[i:i+1] for out in outputs], center, scale)
        result.append((k[0], s[0]))
    return result


def submit_single_pose_calls(model, images, boxes, executor):
    """Preprocess serially, enqueue independent ORT batch-1 calls in input order.

    RTMLib preprocessing mutates normalization attributes, so it must not run
    concurrently on one model. Batch size and model arithmetic remain unchanged.
    """
    if model.backend!='onnxruntime':raise ValueError('Concurrent pose calls require ONNX Runtime')
    if len(images)!=len(boxes):raise ValueError('Image/box count mismatch')
    prepared=[model.preprocess(image,box) for image,box in zip(images,boxes)]
    def infer(item):
        image,center,scale=item
        k,s=model.postprocess(model.inference(image),center,scale)
        return k[0],s[0]
    return [executor.submit(infer,item) for item in prepared]
