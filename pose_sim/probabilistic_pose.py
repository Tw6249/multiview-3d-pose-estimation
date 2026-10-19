"""Experimental PMPose-S adapter; does not change the online/default pipeline.

Input: current BGR exposure and selected xyxy person box in supplied-image px.
Output: COCO17 image coordinates, predicted OKS, crop-presence, visibility and
normalized spatial mass. All probabilities remain uncalibrated in our domain.
No segmentation, reference labels, history or commands. Invalid outputs raise;
the consumer must preserve unknown/expired observations, never reuse old ones.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import os
import subprocess
import sys

import numpy as np

from .geometry import JOINTS
from .sensors import ExposureFrame

REVISION = '49a070a6f8396147323b1c4959474077dbe2ce8e'
WEIGHT_SHA256 = '7255986ab80a4e2c7b60f135fd4b0ffb9438991f21ac1c23a7c54cd1ecf79bee'


@dataclass(frozen=True)
class ProbabilisticPose:
    camera_id: str
    exposure_timestamp_ns: int
    clock_domain: str
    keypoints_px: np.ndarray
    predicted_oks: np.ndarray
    presence: np.ndarray
    visibility: np.ndarray
    spatial_mass: np.ndarray
    center_px: np.ndarray
    scale_px: np.ndarray

    def __post_init__(self):
        from .sensors import nanoseconds
        nanoseconds(self.exposure_timestamp_ns)
        if not self.camera_id or not self.clock_domain:
            raise ValueError('Camera and clock identity required')
        shapes = dict(keypoints_px=(17, 2), predicted_oks=(17,), presence=(17,),
                      visibility=(17,), spatial_mass=(17, 64, 48),
                      center_px=(2,), scale_px=(2,))
        for name, shape in shapes.items():
            a = np.array(getattr(self, name), dtype=np.float32, copy=True)
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError('Invalid ' + name)
            if name in ('predicted_oks', 'presence', 'visibility', 'spatial_mass') and (np.any(a < 0) or np.any(a > 1)):
                raise ValueError('Out-of-range ' + name)
            a.flags.writeable = False
            object.__setattr__(self, name, a)
        if np.any(self.scale_px <= 0) or not np.allclose(self.spatial_mass.sum((1, 2)), 1, atol=1e-4, rtol=0):
            raise ValueError('Invalid crop scale or spatial mass normalization')

    def pixels_to_heatmap(self, pixels):
        """Official OKSArgMax/topdown inverse: x right, y down; no clipping."""
        a = np.asarray(pixels, dtype=float)
        if a.ndim == 0 or a.shape[-1] != 2 or not np.isfinite(a).all():
            raise ValueError('Expected finite (..., 2) pixels')
        return (a - self.center_px + .5 * self.scale_px) / self.scale_px * [47, 63]

    @property
    def semantics(self):
        return dict(model_sha256=WEIGHT_SHA256, joint_names=JOINTS,
                    representation='normalized_joint_xy_mass', calibrated_for_domain=False,
                    presence_event='joint_inside_model_crop', visibility_event='joint_visible',
                    score='predicted_oks_not_visibility', coordinate_frame='supplied_image_xy_pixels')


def extract_prediction(sample, frame):
    """Read actual MultiHead outputs; reject missing fields, never substitute."""
    frame.validate()
    p = sample.pred_instances
    required = ('keypoints', 'keypoints_oks', 'keypoints_probs', 'keypoints_visible')
    if any(not hasattr(p, name) for name in required) or not hasattr(sample, 'pred_fields'):
        raise ValueError('Missing genuine PMPose outputs')
    def scalar(name):
        a = np.asarray(getattr(p, name))
        if a.shape != (1, 23):
            raise ValueError('Expected merged COCO/MPII/AIC 23-joint head')
        return a[0, :17]
    k = np.asarray(p.keypoints)
    if k.shape != (1, 23, 2):
        raise ValueError('Invalid PMPose keypoint layout')
    h = sample.pred_fields.heatmaps
    if hasattr(h, 'detach'):
        h = h.detach().cpu().numpy()
    if np.asarray(h).shape != (23, 64, 48):
        raise ValueError('Invalid PMPose heatmap layout')
    return ProbabilisticPose(frame.camera_id, frame.exposure_timestamp_ns, frame.clock_domain,
        k[0, :17], scalar('keypoints_oks'), scalar('keypoints_probs'),
        scalar('keypoints_visible'), h[:17], sample.metainfo['input_center'], sample.metainfo['input_scale'])


class PMPoseSmall:
    def __init__(self, repository, checkpoint):
        repo, checkpoint = Path(repository).resolve(), Path(checkpoint).resolve()
        revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != REVISION or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != WEIGHT_SHA256:
            raise ValueError('Unpinned PMPose source/weights')
        if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
            raise ValueError('Official source has local modifications')
        sys.path.insert(0, str(repo))
        import torch
        from mmengine.config import Config
        from mmpose.apis import init_model, inference_topdown
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for this experiment')
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cfg = Config.fromfile(str(repo / 'mmpose/configs/ProbMaskPose/PMPose-s-1.0.0.py'))
        # Explicit no-mask ablation: do not invent a segmentation mask.
        for transform in cfg.test_dataloader.dataset.pipeline:
            if transform['type'] == 'MaskBackground':
                transform['continue_on_failure'] = True
        cfg.model.test_cfg.output_heatmaps = True
        old = Path.cwd()
        try:
            os.chdir(repo / 'mmpose')  # official relative metainfo files
            self.model = init_model(cfg, str(checkpoint), device='cuda:0')
        finally:
            os.chdir(old)
        # init_model's load is non-strict. Verify all tensors separately.
        ckpt = torch.load(checkpoint, map_location='cpu')
        self.model.load_state_dict(ckpt['state_dict'], strict=True)
        names = self.model.dataset_meta['keypoint_id2name']
        if [names[i] for i in range(17)] != JOINTS or type(self.model.head).__name__ != 'MultiHead':
            raise ValueError('Unsupported PMPose metadata/head')
        self.inference = inference_topdown
        from mmengine.dataset import Compose, pseudo_collate
        self.pipeline = Compose(self.model.cfg.test_dataloader.dataset.pipeline)
        self.collate = pseudo_collate
        from importlib.metadata import version
        self.metadata = dict(model='PMPose-S-1.0.0', sha256=WEIGHT_SHA256, source_revision=revision,
            torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
            mask_mode='none_continue_on_failure', flip_test=True, dtype='float32', calibrated=False,
            batch_mode='current_camera_crops', tf32=False, cpu_threads=4,
            dependencies={name: version(name) for name in ('numpy', 'scipy', 'opencv-python', 'mmcv', 'mmengine', 'mmpretrain', 'sparsemax')})

    def predict_batch(self, frames_and_boxes):
        """Batch distinct current camera crops with the official test pipeline."""
        import torch
        if not frames_and_boxes:
            return []
        batch = []
        for frame, bbox in frames_and_boxes:
            frame.validate()
            box = np.asarray(bbox, dtype=np.float32)
            if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
                raise ValueError('Invalid selected crop')
            data = dict(img=frame.image_bgr, bbox=box[None], segmentation=None,
                        bbox_score=np.ones(1, dtype=np.float32))
            data.update(self.model.dataset_meta)
            batch.append(self.pipeline(data))
        with torch.no_grad():
            predictions = self.model.test_step(self.collate(batch))
        if len(predictions) != len(frames_and_boxes):
            raise ValueError('Missing batched predictions')
        return [extract_prediction(p, frame) for p, (frame, _) in zip(predictions, frames_and_boxes)]

    def __call__(self, frame: ExposureFrame, bbox_xyxy):
        frame.validate()
        box = np.asarray(bbox_xyxy, dtype=np.float32)
        if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
            raise ValueError('A valid current selected person crop is required')
        result = self.inference(self.model, frame.image_bgr, bboxes=box[None], masks=None)
        if len(result) != 1:
            raise ValueError('Expected exactly one current crop prediction')
        return extract_prediction(result[0], frame)
