"""Experimental COCO17 head comparison on an already selected current crop.

Input/output pixels retain camera/exposure identity; no box or person selection.
Unknown views stay unknown; auxiliary landmarks are unchanged. No robot output.
"""
import copy
import hashlib
import numpy as np
from .geometry import JOINTS
from .inference import select_landmarks


def replace_body(row, outputs, cameras, model_sha):
    if row['joint_names'][:17] != JOINTS:
        raise ValueError('Expected COCO17 body prefix')
    out=copy.deepcopy(row)
    for cid,(keypoints,scores) in outputs.items():
        view=out['views'][cid]
        if view.get('keypoints_px') is None or view.get('person_bbox_xyxy_px') is None:
            raise ValueError('No selected current crop')
        k=np.asarray(keypoints);s=np.asarray(scores)
        if k.shape!=(17,2) or s.shape!=(17,) or not np.isfinite(k).all() or not np.isfinite(s).all():
            raise ValueError('Malformed body prediction')
        cam=cameras[cid];k,s,_=select_landmarks(k,s,list(range(17)),cam['width'],cam['height'])
        view['keypoints_px'][:17]=k.tolist();view['scores'][:17]=s.tolist()
        all_k=np.asarray(view['keypoints_px'])
        view['outside_image_landmarks']=np.flatnonzero((all_k[:,0]<0)|(all_k[:,0]>=cam['width'])|(all_k[:,1]<0)|(all_k[:,1]>=cam['height'])).tolist()
    out['perception_signature']=hashlib.sha256((row['perception_signature']+'/body-head/'+model_sha).encode()).hexdigest()
    return out


def load_head(mode):
    """Use installed RTMLib official URL; pin returned checksum in run manifest."""
    import onnxruntime as ort
    import rtmlib
    from rtmlib.tools.pose_estimation.rtmpose import RTMPose
    from .runtime import preload_cuda
    from pathlib import Path
    ort.set_default_logger_severity(3);preload_cuda(ort)
    spec=rtmlib.Body.MODE[mode]
    model=RTMPose(spec['pose'],model_input_size=spec['pose_input_size'],device='cuda')
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    options.add_session_config_entry('session.intra_op.allow_spinning','0')
    model.session=None
    model.session=ort.InferenceSession(model.onnx_model,sess_options=options,providers=[('CUDAExecutionProvider',{'use_tf32':'0'})])
    if model.session.get_providers()[0]!='CUDAExecutionProvider':raise RuntimeError('CUDA required')
    model.session.disable_fallback()
    metadata=dict(mode=mode,url=spec['pose'],input_size=list(spec['pose_input_size']),
        sha256=hashlib.sha256(Path(model.onnx_model).read_bytes()).hexdigest(),providers=model.session.get_providers(),
        onnxruntime=ort.__version__,cuda_use_tf32=False,intra_op_threads=4,inter_op_threads=1)
    import json
    pinned=json.loads((Path(__file__).resolve().parents[1]/'assets/model_checksums.json').read_text())['experimental_body_heads']
    if metadata['sha256']!=pinned[mode]:raise ValueError('Experimental body head checksum mismatch')
    for _ in range(3):model(np.zeros((1024,1024,3),np.uint8),bboxes=[np.array([10,10,900,900],np.float32)])
    return model,metadata
