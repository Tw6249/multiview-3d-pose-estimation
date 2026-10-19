"""Offline terminal recovery comparison on immutable source references."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motion_control.humanoid_replay import replay
from motion_control.recovery_rollout import load_config


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run(sources,output,config=None):
    output=Path(output).resolve();cfg=config or load_config()
    if output.exists():raise FileExistsError('Preserve prior evidence')
    output.mkdir(parents=True)
    freeze=dict(configuration=cfg,source_sha256={str(s):digest(s/'humanoid_reference.jsonl') for s in sources},
        code_sha256={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'motion_control'/n for n in
            ('recovery_rollout.py','humanoid_replay.py','stop_transition.py','twist.py','g1_command.py','tracking_metrics.py')]+[Path(__file__).resolve()]},
        scope='Known scene regression. Offline nominal branch search; no future source frames or hardware commands. Original source targets retained.')
    (output/'freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8');cases=[]
    keys=('source_reference_leg_rmse_rad','root_path_error_p95_m','tracking_ticks','accepted_frames','passed','stopping_passed','balance_passed')
    for source in sources:
        name=source.parent.name if source.name in ('baseline','integrated_displacement') else source.name
        folder=output/name;folder.mkdir();methods={}
        for mode in ('support_aware','rollout'):
            report=replay(source,video=False,save_logs=False,output_name=str(folder/mode),transition_mode=mode,
                support_diagnostics=True,recovery_config=cfg)
            methods[mode]={k:report[k] for k in keys}
            p=report['recovery_rollout']
            methods[mode]['planning']=None if p is None else {k:v for k,v in p.items() if k!='candidates'}
            methods[mode]['standing_stop_audit']=report['standing_stop_audit']
            print(name,mode,{k:report[k] for k in keys},
                  None if p is None else {k:p.get(k) for k in ('accepted','selected','wall_ms')},flush=True)
        unchanged=all(methods['support_aware'][k]==methods['rollout'][k] for k in keys[:4])
        if not unchanged:raise AssertionError('Terminal search altered prior tracking or source coverage')
        cases.append(dict(id=name,tracking_unchanged_before_terminal_stop=unchanged,methods=methods))
    report=dict(freeze=freeze,cases=cases)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--sources',nargs='+',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--device',choices=['cpu','cuda'])
    args=parser.parse_args();cfg=load_config()
    if args.device:cfg['device']=args.device
    run(args.sources,args.output,cfg)
