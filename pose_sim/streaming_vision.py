"""Warm CUDA front end and causal RGB-to-state packet processor; no file replay assumptions."""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import hashlib
import json
import time
import numpy as np
from .runtime import create_body,warmup_body
from .inference import pose_batch,select_landmarks,submit_single_pose_calls,infer_person_candidates,complete_selected_candidates
from .landmarks import SCHEMAS
from .multiview_temporal import MultiViewTemporal
from .human_motion import HumanMotionEstimator
from .person_selection import CrossViewPersonSelector


class StreamingVision:
    def __init__(self,calibration,geometry_config,vision_config,state_config=None,detector_workers=1,pose_workers=1,fusion_config=None):
        if detector_workers not in (1,2,4) or pose_workers not in (1,2,4):raise ValueError('Use one, two or four inference workers')
        self.cameras=calibration['cameras'];self.config=vision_config
        self.body,self.runtime=create_body(vision_config['mode'],'cuda',model_kind=vision_config['model_kind'],
            detector=vision_config['detector'],detector_postprocess=vision_config['detector_postprocess'])
        self.schema,self.names,self.indices=SCHEMAS[vision_config['model_kind']]
        self.signature=hashlib.sha256(json.dumps({k:v['sha256'] for k,v in self.runtime['models'].items()},sort_keys=True).encode()).hexdigest()
        if vision_config.get('cross_view_person_selection',False):
            self.signature=hashlib.sha256((self.signature+'cross_view_person_selection_v1').encode()).hexdigest()
        first=next(iter(self.cameras.values()))
        self.warmup_seconds=warmup_body(self.body,first['width'],first['height'],3,len(self.cameras))
        self.pool=ThreadPoolExecutor(max_workers=detector_workers) if detector_workers>1 else None
        self.pose_pool=ThreadPoolExecutor(max_workers=pose_workers) if pose_workers>1 else None
        self.stage_timings=deque(maxlen=4096);self.last_detection_timing={}
        fusion_config=fusion_config or {}
        self.fusion=MultiViewTemporal(calibration,geometry_config,
            association=fusion_config.get('association',True),
            structured=fusion_config.get('structured_reprojection',False),
            auxiliary=fusion_config.get('auxiliary_association',True),
            history=fusion_config.get('association_history',False),
            observation_consistency=fusion_config.get('observation_consistency',False),
            window_config=fusion_config.get('joint_window'))
        self.state=HumanMotionEstimator(state_config);self.previous=None
        self.person_selector=CrossViewPersonSelector(self.cameras,geometry_config) if vision_config.get('cross_view_person_selection',False) else None
        self.geometry_config=geometry_config

    def infer_candidates(self,cid,image,boxes,body_only=False):
        """Bounded fresh-image inference, shared by online and sparse cache paths."""
        return infer_person_candidates(self.body,self.config['model_kind'],self.indices,self.cameras[cid],image,boxes,body_only)

    def detect(self,packet):
        begin=time.perf_counter()
        packet.validate()
        row=dict(frame_id=packet.frame_id,timestamp_ns=packet.timestamp_ns,source_id='single_person',
            landmark_schema_id=self.schema,joint_names=self.names,perception_signature=self.signature,views={})
        items=list(packet.frames.items());images=[f.image_bgr for _,f in items]
        for c,f in items:
            if c not in self.cameras or f.image_bgr.shape[:2]!=(self.cameras[c]['height'],self.cameras[c]['width']):
                raise ValueError('Image shape/camera differs from calibration')
        boxes=list(self.pool.map(self.body.det_model,images)) if self.pool else [self.body.det_model(im) for im in images]
        detector_done=time.perf_counter()
        pending=[]
        for (c,f),im,b in zip(items,images,boxes):
            row['views'][c]=dict(timestamp_ns=f.exposure_timestamp_ns,num_people_detected=len(b),keypoints_px=None,scores=None)
            if len(b)==1:pending.append((c,im,b[0]))
        futures=submit_single_pose_calls(self.body.body_pose_model,[v[1] for v in pending],[v[2] for v in pending],self.pose_pool) \
            if self.config['model_kind']=='hybrid' and self.pose_pool else None
        results=pose_batch(self.body.pose_model,[v[1] for v in pending],[v[2] for v in pending])
        for i,((c,im,box),(k,s)) in enumerate(zip(pending,results)):
            row['views'][c]['person_bbox_xyxy_px'] = np.asarray(box[:4],float).tolist()
            if self.config['model_kind']=='hybrid':
                # Preserve the detail model's attachment roots before replacing
                # its body landmarks. They are observations, not ground truth.
                row['views'][c]['auxiliary_model_anchors']={
                    n:dict(keypoint_px=k[j].tolist(),score=float(s[j]))
                    for n,j in (('left_ankle',15),('right_ankle',16))
                    if np.isfinite(k[j]).all() and np.isfinite(s[j])
                    and 0<=k[j][0]<self.cameras[c]['width'] and 0<=k[j][1]<self.cameras[c]['height']}
                if futures:bk,bs=futures[i].result()
                else:
                    bk,bs=self.body.body_pose_model(im,bboxes=[box]);bk,bs=bk[0],bs[0]
                k[:17]=bk;s[:17]=bs
            cam=self.cameras[c];k,s,outside=select_landmarks(k,s,self.indices,cam['width'],cam['height'])
            if np.isfinite(k).all() and np.isfinite(s).all():
                row['views'][c].update(keypoints_px=k.tolist(),scores=s.tolist(),outside_image_landmarks=outside)
        self.last_detection_timing=dict(detector_ms=(detector_done-begin)*1000,
                                       pose_ms=(time.perf_counter()-detector_done)*1000,
                                       person_selection_ms=0.)
        if self.person_selector:
            if self.person_selector.cameras is not self.cameras:
                self.person_selector=CrossViewPersonSelector(self.cameras,self.geometry_config)
            for (c,f),im,b in zip(items,images,boxes):
                if 2<=len(b)<=4:row['views'][c]['person_candidates']=self.infer_candidates(c,im,b,body_only=True)
            selection_begin=time.perf_counter()
            self.last_detection_timing['pose_ms']=(selection_begin-detector_done)*1000
            row=self.person_selector.update(row)
            self.last_detection_timing['person_selection_ms']=(time.perf_counter()-selection_begin)*1000
            completion_begin=time.perf_counter()
            row=complete_selected_candidates(self.body,self.config['model_kind'],self.indices,self.cameras,
                {c:f.image_bgr for c,f in items},row)
            self.last_detection_timing['pose_ms']+=(time.perf_counter()-completion_begin)*1000
        return row

    def process(self,packet):
        detection=self.detect(packet);begin=time.perf_counter();pose,assigned=self.fusion.update(detection,self.previous)
        fused=time.perf_counter()
        self.previous=self.state.update(pose,self.cameras,assigned)
        self.stage_timings.append(dict(**self.last_detection_timing,fusion_ms=(fused-begin)*1000,
                                      state_ms=(time.perf_counter()-fused)*1000))
        # A synthetic calibration may not be silently used on RGB/real streams.
        self.previous['uncertainty_calibration']=dict(valid_for_domain=False,domain='unvalidated_rgb',guarantee_claimed=False)
        return self.previous

    def close(self):
        if self.pool:self.pool.shutdown(wait=True)
        if self.pose_pool:self.pose_pool.shutdown(wait=True)
