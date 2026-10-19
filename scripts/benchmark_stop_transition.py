"""Paired stop-transition simulations on identical source files; no copies/labels."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motion_control.humanoid_replay import replay
from motion_control.stop_transition import load_config


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run(sources,output,config=None,dropout=None,modes=('legacy','timed','support_aware')):
    if output.exists():raise FileExistsError('Keep previous transition evidence')
    output.mkdir(parents=True);cfg=copy.deepcopy(config or load_config())
    freeze=dict(configuration=cfg,dropout=dropout,modes=list(modes),hardware_enabled=False,
        source_sha256={str(s):sha(s/'humanoid_reference.jsonl') for s in sources},
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'motion_control'/n for n in ('humanoid_replay.py','stop_transition.py','support_feedback.py','centroidal_feasibility.py','tracking_metrics.py','twist.py','root_tracking.py')]+[Path(__file__).resolve()]},
        decision='No default promotion without preserving original-source tracking/coverage and improving upright stopping. Contact feedback is robot-only; source files/targets unchanged.')
    (output/'freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8');cases=[]
    for source in sources:
        name=source.parent.name if source.name in ('integrated_displacement','baseline') else source.name
        folder=output/name;folder.mkdir();methods={}
        for mode in modes:
            methods[mode]=replay(source,video=False,output_name=str((folder/mode).resolve()),save_logs=False,
                transition_mode=mode,transition_config=cfg,support_diagnostics=True,dropout=dropout)
            m=methods[mode]
            print(name,mode,{k:m[k] for k in ('passed','balance_passed','stopping_passed','source_reference_leg_rmse_rad','root_path_error_p95_m')},flush=True)
        unchanged=None
        if 'legacy' in methods and not cfg['apply_to_missing']:
            keys=('source_reference_leg_rmse_rad','root_path_error_p95_m','tracking_ticks','accepted_frames')
            unchanged=all(m[k]==methods['legacy'][k] for m in methods.values() for k in keys)
            if not unchanged:raise AssertionError('Terminal-only stop altered earlier source tracking')
        cases.append(dict(id=name,tracking_unchanged_before_terminal_stop=unchanged,methods=methods))
    report=dict(freeze=freeze,cases=cases)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sources',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--apply-to-missing',action='store_true');p.add_argument('--dropout',nargs=2,type=float)
    a=p.parse_args();cfg=load_config();cfg['apply_to_missing']=a.apply_to_missing;run(a.sources,a.output,cfg,a.dropout)
