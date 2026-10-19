"""Reproduce the modular vision pipeline against existing RGB without copying it."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pose_sim.pipeline import detect,reconstruct
from pose_sim.human_state import estimate_session
from pose_sim.vision_metrics import evaluate


def prepare_models(cfg):
    from pose_sim.runtime import create_body
    body,_=create_body(cfg['mode'],'cuda',model_kind=cfg['model_kind'],detector=cfg['detector'])
    model=body.det_model.onnx_model
    del body
    robot_python=json.loads((ROOT/'.robot-runtime.json').read_text())['python']
    subprocess.run([robot_python,str(ROOT/'scripts/prepare_vision_models.py'),str(model)],check=True)


def run(source,output,profile=ROOT/'config/vision.json',prepare=False):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Choose a new output directory; preserve the existing baseline')
    cfg=json.loads(Path(profile).read_text())
    if prepare:
        prepare_models(cfg)
    start=time.perf_counter()
    detect(source,mode=cfg['mode'],device='cuda',model_kind=cfg['model_kind'],detector=cfg['detector'],
           batch=cfg['batch'],output=output,detector_postprocess=cfg['detector_postprocess'],
           cross_view_person_selection=cfg.get('cross_view_person_selection',False))
    reconstruct(source,output)
    estimate_session(source,output,ROOT/cfg['state_config'])
    result=evaluate(source,output)
    result['source_session']=str(source)
    result['input_sha256']={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ('frames.jsonl','cameras.json','config.json')}
    image_digest=hashlib.sha256()
    for row in (source/'frames.jsonl').read_text().splitlines():
        for cid,view in sorted(json.loads(row)['views'].items()):
            image_digest.update(cid.encode());image_digest.update(view['image'].encode())
            image_digest.update(hashlib.sha256((source/view['image']).read_bytes()).digest())
    result['rgb_sequence_sha256']=image_digest.hexdigest()
    result['profile']=cfg
    result['wall_seconds_including_initialization']=time.perf_counter()-start
    result['mean_processing_seconds']=sum((result['inference']['mean_seconds_per_packet_with_image_io'],
                                          result['triangulation']['mean_seconds'],result['state']['mean_seconds']))
    result['offline_processing_groups_per_second']=1/result['mean_processing_seconds']
    result['source_sha256']={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted((ROOT/'pose_sim').glob('*.py'))}
    (output/'run_manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('mean_processing_seconds','offline_processing_groups_per_second')},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',nargs='?',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--profile',default=ROOT/'config/vision.json',type=Path)
    p.add_argument('--prepare-models',action='store_true',help='Use the pinned robot runtime to extract cached detector outputs')
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args()
    if a.prepare_only:prepare_models(json.loads(a.profile.read_text()))
    elif a.source is None or a.output is None:p.error('source and --output are required for inference')
    else:run(a.source,a.output,a.profile,a.prepare_models)
