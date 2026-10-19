"""Reproduce markerless human multiview vision -> full-body G1 imitation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from motion_control.humanoid import retarget_session
from motion_control.humanoid_replay import replay


def run(output,frames=96,video=True,perception_delay_ms=120,control_only=False):
    output=Path(output)
    if not control_only and output.exists():raise FileExistsError('Choose a new generated output directory')
    vision_python=json.loads((ROOT/'.runtime.json').read_text())['python']
    timings={}
    def stage(name,call):
        print('STAGE',name,flush=True);t=time.perf_counter();result=call();timings[name]=time.perf_counter()-t
        return result
    def invoke(args):subprocess.run([vision_python,*map(str,args)],cwd=ROOT,check=True)
    if not control_only:
        vision=json.loads((ROOT/'config/vision.json').read_text())
        stage('prepare_vision_models',lambda:invoke(['scripts/vision_demo.py','--prepare-only']))
        stage('render_rgb',lambda:invoke(['scripts/run_blender.py','--output',output,'--config','config/human_walk.json','--frames',frames,'--render-device','OPTIX']))
        stage('cuda_perception',lambda:invoke(['-m','pose_sim.pipeline','detect',output,'--device','cuda',
            '--model-kind',vision['model_kind'],'--mode',vision['mode'],'--detector-postprocess',vision['detector_postprocess'],'--batch']))
        stage('triangulation',lambda:invoke(['-m','pose_sim.pipeline','reconstruct',output]))
        stage('human_state',lambda:invoke(['-m','pose_sim.human_state',output,'--config',vision['state_config']]))
    # The controller retains current image observations. Predicted/filtered
    # source states have a separate quality contract, awaiting controller validation.
    stage('full_body_retarget',lambda:retarget_session(output))
    result=stage('g1_physics_tracking',lambda:replay(output,video,perception_delay_ms))
    if not control_only:stage('independent_vision_evaluation',lambda:invoke(['-m','pose_sim.vision_metrics',output]))
    result['stage_seconds']=timings
    result['source_sha256']={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('pose_sim','motion_control','scripts','config','tests') for p in sorted((ROOT/folder).glob('*')) if p.suffix in ('.py','.json')}
    result['runtime']={n:json.loads((ROOT/n).read_text()) for n in ('.runtime.json','.robot-runtime.json')}
    (output/'run_manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('passed','balance_passed','imitation_passed','root_path_error_p95_m','source_reference_leg_rmse_rad')},indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--frames',type=int,default=96);p.add_argument('--no-video',action='store_true')
    p.add_argument('--perception-delay-ms',type=float,default=120);p.add_argument('--control-only',action='store_true')
    a=p.parse_args();result=run(a.output,a.frames,not a.no_video,a.perception_delay_ms,a.control_only)
    raise SystemExit(0 if result['passed'] else 1)
