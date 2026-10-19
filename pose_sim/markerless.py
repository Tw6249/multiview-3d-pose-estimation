"""Adapter for source-specific, markerless ONNX landmark models.

No pretrained robot recognizer is bundled. The export contract is deliberately
explicit: RGB float32 /255 stretched to [1,3,H,W], output [1,instances,N,3]
(input-image pixel x,y,score). A graph must include instance detection and bind
its semantic landmark order to the source profile in ONNX custom metadata.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from .pipeline import read_jsonl, write_jsonl
from .robot_source import RobotSource
from .runtime import preload_cuda
from .sensors import validate_calibration


def decode_landmarks(output, count, image_size, input_size):
    output = np.asarray(output)
    if output.ndim != 4 or output.shape[0] != 1 or output.shape[2:] != (count, 3):
        raise ValueError('Expected ONNX landmarks [1,instances,N,3]')
    instances = output.shape[1]
    if instances != 1:
        return None, None, instances
    values = output[0, 0]
    if not np.isfinite(values).all() or np.any(values[:, 2] < 0) or np.any(values[:, 2] > 1):
        raise ValueError('Nonfinite landmarks or scores outside [0,1]')
    width, height = image_size
    input_width, input_height = input_size
    points, scores = [], []
    for x, y, score in values:
        px, py = (float(x)+0.5)*width/input_width-0.5, (float(y)+0.5)*height/input_height-0.5
        if not (0 <= px < width and 0 <= py < height) or score == 0:
            points.append(None)
            scores.append(0.0)
        else:
            points.append([px, py])
            scores.append(float(score))
    return points, scores, instances


class OnnxRobotDetector:
    def __init__(self, source, model, model_sha256, device='cuda'):
        self.source = source
        if device not in ('cpu', 'cuda'):
            raise ValueError('Select cpu or cuda explicitly')
        path = Path(model)
        with path.open('rb') as handle:
            self.sha256 = hashlib.file_digest(handle, 'sha256').hexdigest()
        if self.sha256 != model_sha256:
            raise ValueError('Robot detector checksum mismatch')
        import onnxruntime as ort
        expected = 'CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'
        if expected not in ort.get_available_providers():
            raise RuntimeError(f'{expected} unavailable; refusing silent device fallback')
        if device == 'cuda':
            preload_cuda(ort)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        providers = [(expected, {'use_tf32': '0'})] if device == 'cuda' else [expected]
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        self.providers = self.session.get_providers()
        if not self.providers or self.providers[0] != expected:
            raise RuntimeError('Robot detector activated an unexpected provider; refusing fallback')
        self.session.disable_fallback()
        metadata = self.session.get_modelmeta().custom_metadata_map
        required = {'perception_mode': 'markerless', 'source_model_sha256': source.sha256,
                    'input_encoding': 'rgb_float32_0_1_stretch', 'output_encoding': 'image_xy_score'}
        if any(metadata.get(k) != v for k, v in required.items()):
            raise ValueError('ONNX metadata differs from markerless source/preprocessing contract')
        if json.loads(metadata.get('landmark_names', 'null')) != source.names:
            raise ValueError('ONNX semantic landmark order differs from source profile')
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError('Export one image input and one landmarks output')
        self.input, self.output = inputs[0], outputs[0]
        shape = self.input.shape
        if (self.input.type != 'tensor(float)' or len(shape) != 4 or shape[:2] != [1, 3] or
                any(type(v) is not int or v <= 0 for v in shape[2:])):
            raise ValueError('Export a fixed float32 [1,3,H,W] input')
        if self.output.type != 'tensor(float)':
            raise ValueError('Export float32 landmarks')
        self.input_size = (shape[3], shape[2])

    def detect_image(self, image):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Expected uint8 BGR image')
        rgb = cv2.cvtColor(cv2.resize(image, self.input_size), cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)/255
        output = self.session.run([self.output.name], {self.input.name: tensor})[0]
        return decode_landmarks(output, len(self.source.names), (image.shape[1], image.shape[0]), self.input_size)


def detect(session, source, detector):
    session = Path(session)
    if detector.source.sha256 != source.sha256:
        raise ValueError('Detector and source profile differ')
    calibration = json.loads((session/'cameras.json').read_text(encoding='utf-8'))
    validate_calibration(calibration)
    rows, timings = [], []
    for packet in read_jsonl(session/'frames.jsonl'):
        row = {'frame_id': packet['frame_id'], 'timestamp_ns': packet['timestamp_ns'], 'views': {},
               'joint_names': source.names, 'landmark_schema_id': source.config['landmark_schema_id'],
               'source_id': source.config['source_id'], 'source_model_sha256': source.sha256,
               'detector_model_sha256': detector.sha256, 'perception_mode': 'markerless'}
        for cid, frame in packet['views'].items():
            camera = calibration['cameras'][cid]
            image = cv2.imdecode(np.fromfile(session/frame['image'], dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (camera['height'], camera['width']):
                raise ValueError('Unreadable image or dimensions differ from calibration')
            begin = time.perf_counter()
            points, scores, instances = detector.detect_image(image)
            elapsed = time.perf_counter()-begin
            timings.append(elapsed)
            row['views'][cid] = {'timestamp_ns': frame['timestamp_ns'], 'keypoints_px': points,
                                 'scores': scores, 'num_source_instances': instances, 'inference_seconds': elapsed}
        rows.append(row)
    if not rows:
        raise ValueError('Empty source session')
    write_jsonl(session/'detections.jsonl', rows)
    (session/'source_model.json').write_text(json.dumps(source.config, indent=2), encoding='utf-8')
    (session/'inference_timing.json').write_text(json.dumps({
        'backend': 'onnxruntime_robot_markerless', 'providers': detector.providers,
        'source_model_sha256': source.sha256, 'detector_model_sha256': detector.sha256,
        'image_count': len(timings), 'mean_seconds_per_image': float(np.mean(timings)),
        'note': 'Includes first inference; no warmup. Model accuracy requires source-specific real-data evaluation.'
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    parser.add_argument('--source-config', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-sha256', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    source = RobotSource(json.loads(args.source_config.read_text(encoding='utf-8')))
    detector = OnnxRobotDetector(source, args.model, args.model_sha256, args.device)
    detect(args.session, source, detector)
