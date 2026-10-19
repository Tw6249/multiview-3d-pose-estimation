"""Select and verify the inference device before processing image data."""
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

_DLL_HANDLES = []
_DLL_DIRECTORIES = set()


def preload_cuda(ort):
    # cuDNN loads additional engine DLLs lazily during the first convolution.
    # ORT preloads its direct dependencies; expose all NVIDIA bin directories too.
    if os.name == 'nt':
        root = Path(sys.prefix)/'Lib/site-packages/nvidia'
        bins = [str(p.resolve()) for p in root.glob('*/bin') if p.is_dir()]
        for folder in bins:
            if folder not in _DLL_DIRECTORIES:
                _DLL_HANDLES.append(os.add_dll_directory(folder))
                _DLL_DIRECTORIES.add(folder)
        existing = os.environ.get('PATH', '').split(os.pathsep)
        os.environ['PATH'] = os.pathsep.join(bins+[p for p in existing if p not in bins])
    ort.preload_dlls(directory='')


def create_body(mode='balanced', device='auto', profile_dir=None, model_kind='body17', detector='default', detector_postprocess='onnx'):
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
    if device not in ('auto', 'cpu', 'cuda'):
        raise ValueError(f'Unknown inference device: {device}')
    if detector_postprocess not in ('onnx','cpu'):
        raise ValueError('Unknown detector postprocess')
    if device != 'cpu' and 'CUDAExecutionProvider' in ort.get_available_providers():
        preload_cuda(ort)
    selected = ('cuda' if 'CUDAExecutionProvider' in ort.get_available_providers() else 'cpu') if device == 'auto' else device
    if selected == 'cuda' and 'CUDAExecutionProvider' not in ort.get_available_providers():
        raise RuntimeError('CUDA provider unavailable. Run bootstrap.py --device cuda, or explicitly select --device cpu.')
    import rtmlib
    begin = time.perf_counter()
    if model_kind not in ('body17', 'wholebody', 'hybrid') or detector not in ('default', 'tiny'):
        raise ValueError('Unknown pose model or detector')
    factory = rtmlib.Body if model_kind == 'body17' else rtmlib.Wholebody
    kwargs = {}
    if detector == 'tiny':
        kwargs = dict(det=rtmlib.Body.MODE['lightweight']['det'], det_input_size=(416, 416))
    body = factory(mode=mode, backend='onnxruntime', device=selected, to_openpose=False, **kwargs)
    if model_kind == 'hybrid':
        from rtmlib.tools.pose_estimation.rtmpose import RTMPose
        spec=rtmlib.Body.MODE['balanced']
        body.body_pose_model=RTMPose(spec['pose'],model_input_size=spec['pose_input_size'],
                                    backend='onnxruntime',device=selected,to_openpose=False)
    models = {}
    expected_hashes = json.loads((Path(__file__).resolve().parents[1]/'assets/model_checksums.json').read_text())
    expected = 'CUDAExecutionProvider' if selected == 'cuda' else 'CPUExecutionProvider'
    for name in ('det_model', 'pose_model') + (('body_pose_model',) if model_kind=='hybrid' else ()):
        model = getattr(body, name)
        providers = model.session.get_providers()
        if providers[0] != expected:
            raise RuntimeError(f'{name} requested {expected}, but activated {providers}; refusing silent CPU fallback')
        source_path = Path(model.onnx_model)
        source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        conversion = None
        if name == 'det_model' and detector_postprocess == 'cpu':
            derived = source_path.parent/'cpu_nms'/source_path.name
            if not derived.exists() or not derived.with_suffix('.json').exists():
                raise RuntimeError('Prepare detector with pinned robot Python: scripts/prepare_vision_models.py '+str(source_path))
            conversion = json.loads(derived.with_suffix('.json').read_text())
            if conversion['source_sha256'] != source_sha or conversion['sha256'] != hashlib.sha256(derived.read_bytes()).hexdigest():
                raise RuntimeError('Derived detector model checksum mismatch')
            model.onnx_model = str(derived)
        # RTMLib does not expose SessionOptions. Replace its default session with
        # a bounded thread pool; default per-model pools oversubscribe this host.
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        if profile_dir is not None:
            Path(profile_dir).mkdir(parents=True, exist_ok=True)
            options.enable_profiling = True
            options.profile_file_prefix = str(Path(profile_dir)/name)
        providers = [(expected, {'use_tf32': '0'})] if selected == 'cuda' else [expected]
        model.session = None
        model.session = ort.InferenceSession(model.onnx_model, sess_options=options, providers=providers)
        providers = model.session.get_providers()
        if providers[0] != expected:
            raise RuntimeError(f'{name} requested {expected}, but activated {providers}; refusing silent CPU fallback')
        model.session.disable_fallback()
        path = Path(model.onnx_model)
        with path.open('rb') as f:
            sha256 = hashlib.file_digest(f, 'sha256').hexdigest()
        key = mode if model_kind == 'body17' and detector == 'default' else f'{model_kind}/{mode}/{detector}'
        expected_hash = expected_hashes.get(key, {}).get(name)
        if expected_hash is not None and source_sha != expected_hash:
            raise RuntimeError(f'{name} checksum mismatch for {mode}; expected {expected_hash}, found {sha256}')
        models[name] = {'providers': providers, 'onnx_file': path.name, 'sha256': sha256,
                        'source_sha256': source_sha, 'conversion': conversion}
        if conversion is not None:
            from .inference import CpuNmsDetector
            body.det_model = CpuNmsDetector(model,conversion)
    libraries = {}
    for package in ('onnxruntime', 'onnxruntime-gpu', 'nvidia-cudnn-cu12', 'nvidia-cuda-runtime-cu12', 'nvidia-cublas-cu12'):
        try:
            libraries[package] = version(package)
        except PackageNotFoundError:
            continue
    return body, {'device': selected, 'model_kind': model_kind, 'detector': detector, 'detector_postprocess': detector_postprocess,
                  'onnxruntime_version': ort.__version__, 'libraries': libraries, 'models': models,
                  'intra_op_threads': 4, 'inter_op_threads': 1, 'thread_spinning': False,
                  'cuda_use_tf32': False if selected == 'cuda' else None,
                  'initialization_seconds': time.perf_counter()-begin}


def warmup_body(body, width, height, count, batch_size=1):
    begin = time.perf_counter()
    image = np.zeros((height, width, 3), dtype=np.uint8)
    # Always warm up pose inference too, even when the blank detector input has no person.
    boxes = np.array([[0, 0, width-1, height-1]], dtype=np.float32)
    for _ in range(count):
        body.det_model(image)
        if batch_size == 1:
            body.pose_model(image, bboxes=boxes)
        else:
            from .inference import pose_batch
            pose_batch(body.pose_model,[image]*batch_size,[boxes[0]]*batch_size)
        if hasattr(body,'body_pose_model'):
            body.body_pose_model(image,bboxes=boxes)
    return time.perf_counter()-begin
