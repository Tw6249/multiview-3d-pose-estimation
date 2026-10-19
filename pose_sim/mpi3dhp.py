"""MPI-INF-3DHP conversion contracts (not part of perception).

Input: published Skeletool pinhole calibration in mm, annot3 in camera mm,
zero-based frames/cameras and 25 Hz video. Output: right-handed Z-up world m,
OpenCV camera coordinates, dataset-relative integer ns. No clock accuracy or
contact ground truth is invented. Malformed calibration/frame maps fail closed.
"""
import re
import numpy as np
from scipy.io import loadmat
from .sensors import validate_calibration

WORLD_ROTATION = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
BODY_MAP = dict(left_shoulder=9, right_shoulder=14, left_elbow=10, right_elbow=15,
                left_wrist=11, right_wrist=16, left_hip=18, right_hip=23,
                left_knee=19, right_knee=24, left_ankle=20, right_ankle=25)


def parse_calibration(text, camera_ids, width=1024, height=1024):
    parts = re.split(r'(?m)^name\s+(\d+)\s*$', text)
    if not text.startswith('Skeletool Camera Calibration File V1.0') or len(set(camera_ids)) != len(camera_ids):
        raise ValueError('Unsupported calibration / duplicate cameras')
    original = {}
    cameras = {}
    for name, block in zip(parts[1::2], parts[2::2]):
        cid = int(name)
        if cid not in camera_ids:
            continue
        def array(key):
            match = re.search(r'(?m)^\s*'+key+r'\s+(.+)$', block)
            if not match:
                raise ValueError('Missing calibration '+key)
            return np.fromstring(match[1], sep=' ')
        if array('animated').tolist() != [0.] or array('radial').tolist() != [0.]:
            raise ValueError('Moving/distorted cameras need a separate adapter')
        k = array('intrinsic').reshape(4, 4)[:3, :3]
        extrinsic = array('extrinsic').reshape(4, 4)
        if not np.allclose(extrinsic[3], [0, 0, 0, 1]):
            raise ValueError('Invalid rigid transform')
        resolution = array('size')
        original[cid] = dict(K=k.copy(), R=extrinsic[:3, :3], t=extrinsic[:3, 3], resolution=resolution)
        # cv2.resize pixel-center convention: u'=(u+.5)*scale-.5.
        sx, sy = width/resolution[0], height/resolution[1]
        scale = np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1]])
        cameras[str(cid)] = dict(K=(scale@k).tolist(),
            R_world_to_camera=(extrinsic[:3, :3]@WORLD_ROTATION.T).tolist(),
            t_world_to_camera_m=(extrinsic[:3, 3]/1000).tolist(),
            distortion=[0.]*5, width=width, height=height)
    if set(original) != set(camera_ids):
        raise ValueError('Missing selected camera calibration')
    result = dict(schema_version=1, units='metre', world_axes='X=provider X, Y=-provider Z, Z=provider Y',
                  clock_domain='mpi3dhp_frame_index', cameras=cameras)
    validate_calibration(result)
    return result, original


def reference_rows(annotation_path, original, start, count, fps=25, stride=1):
    """Reference-only conversion; never called from RGB inference.

    stride>1 takes every stride-th source frame from a fps*stride Hz sequence;
    timestamps stay on the source frame clock.
    """
    if min(start, count-1) < 0 or fps != 25 or type(stride) is not int or stride < 1:
        raise ValueError('This validated subset requires nonnegative indices and 25 Hz')
    stop = start+(count-1)*stride+1
    data = loadmat(annotation_path, variable_names=['cameras', 'frames', 'annot3', 'annot2'])
    frames = data['frames'].ravel().astype(int)
    cameras = data['cameras'].ravel().astype(int).tolist()
    if not np.array_equal(frames, np.arange(len(frames))) or stop > len(frames):
        raise ValueError('Unknown frame correspondence or truncated annotations')
    worlds = []
    reprojections = {}
    for cid, camera in original.items():
        index = cameras.index(cid)
        xyz = data['annot3'][index, 0].reshape(-1, 28, 3)[start:stop:stride]
        uv = data['annot2'][index, 0].reshape(-1, 28, 2)[start:stop:stride]
        world = ((xyz-camera['t'])@camera['R'])@WORLD_ROTATION.T/1000
        worlds.append(world)
        projected = xyz@camera['K'].T
        projected = projected[..., :2]/projected[..., 2:]
        error = np.linalg.norm(projected-uv, axis=-1)
        reprojections[str(cid)] = dict(median_px=float(np.median(error)), p95_px=float(np.percentile(error, 95)),
                                       max_px=float(np.max(error)))
    discrepancy = max(float(np.max(np.linalg.norm(w-worlds[0], axis=-1))) for w in worlds)
    if not np.isfinite(worlds).all() or discrepancy > .001:
        raise ValueError('Camera reference transforms disagree by more than 1 mm')
    # A fixed camera's reference defines world positions; do not select/fit to predictions.
    points = worlds[0][:, list(BODY_MAP.values())]
    rows = [dict(frame_id=i, timestamp_ns=round((start+i*stride)*1e9/(fps*stride)),
                 joint_names=list(BODY_MAP), joints_world_m=p.tolist(),
                 reference_type='markerless_multiview_annot3', source_frame_index=start+i*stride)
            for i, p in enumerate(points)]
    return rows, dict(camera_reference_max_disagreement_mm=discrepancy*1000,
                      annotation_reprojection_original_px=reprojections,
                      frame_mapping='annot.frames = zero-based decoded video index')
