"""Compare identical source motions in G1 dynamics; keep original scoring targets."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pose_sim.pipeline import read_jsonl,write_jsonl
from motion_control.offline_reference import regularize
from motion_control.humanoid_replay import replay
from motion_control.twist import TwistPlant
from motion_control.reference_trajectory import load_config


def run(sources,output,weight=.08):
    if output.exists():raise FileExistsError('Preserve previous tracking audit')
    output.mkdir(parents=True);plant=TwistPlant();cases=[]
    code=hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in sorted((ROOT/'motion_control').glob('*.py')))).hexdigest()
    freeze=dict(implementation_sha256=code,weight=weight,candidate='offline regularized references + existing rate limiter',
        experimental_jerk_mode='not selected: development leg-phase regression and position-limit solver failures',
        source_files={str(p):hashlib.sha256((p/'humanoid_reference.jsonl').read_bytes()).hexdigest() for p in sources},
        trajectory_configuration=load_config(),hardware_enabled=False)
    (output/'freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8')
    for source in sources:
        case=output/source.name;case.mkdir();modified=case/'offline_input';modified.mkdir()
        rows=list(read_jsonl(source/'humanoid_reference.jsonl'));optimized=regularize(rows,plant.limits,weight)
        write_jsonl(modified/'humanoid_reference.jsonl',optimized)
        (modified/'retargeting.json').write_text(json.dumps(dict(source=str(source),source_sha256=freeze['source_files'][str(source)],
            operation='offline regularization using future estimates, original joint/path references retained',ground_truth_used=False),indent=2),encoding='utf-8')
        results={}
        for method,folder in (('baseline',source),('offline_regularized',modified)):
            try:results[method]=replay(folder,video=False,output_name=str((case/method).resolve()),save_logs=False)
            except ValueError as error:results[method]=dict(passed=False,error=str(error))
        cases.append(dict(id=source.name,methods=results))
        print(source.name,{m:{k:r.get(k) for k in ('passed','balance_passed','stopping_passed','source_reference_leg_rmse_rad','root_path_error_p95_m')} for m,r in results.items()},flush=True)
    report=dict(freeze=freeze,cases=cases,limitation='Motion tracking simulation only; original estimated reference is not motion-capture truth. Offline optimization uses future estimates; no live latency or hardware claim.')
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sources',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.sources,a.output)
