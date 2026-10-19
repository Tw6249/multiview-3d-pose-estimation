"""Paired source-heading/robot-heading interface audit on unchanged observations."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motion_control.root_tracking import load_config
from motion_control.humanoid_replay import replay


def run(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve old root-frame evidence')
    output.mkdir(parents=True)
    previous=Path('outputs/recovery_rollout/final_validation');old=json.loads((previous/'report.json').read_text())
    sources=[Path(p) for p in old['freeze']['source_sha256']]
    cfg=load_config();freeze=dict(configuration=cfg,sources_sha256=old['freeze']['source_sha256'],
        code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [ROOT/'motion_control'/n for n in ('root_tracking.py','humanoid_replay.py','twist.py')]+[Path(__file__).resolve()]},
        scope='Coordinate-contract correction, no gain tuning; known-scene regressions, not unseen evaluation. Same source timestamp, joint/pose references, phase and acceptance.')
    (output/'freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8');cases=[]
    keys=('accepted_frames','source_frames','tracking_ticks','source_reference_leg_rmse_rad','root_path_error_p95_m','balance_passed','stopping_passed','passed','root_feedforward_frame_error_m_s','standing_stop_audit','balance_by_stage')
    for source in sources:
        if hashlib.sha256((source/'humanoid_reference.jsonl').read_bytes()).hexdigest()!=freeze['sources_sha256'][str(source)]:raise ValueError('Source changed')
        name=source.parent.name if source.name in ('baseline','integrated_displacement') else source.name
        folder=output/name;folder.mkdir();methods={}
        for mode in ('legacy_unrotated','source_heading'):
            cfg=load_config();cfg['velocity_frame']=mode
            report=replay(source,video=False,save_logs=False,output_name=str(folder/mode),transition_mode='support_aware',root_tracking_config=cfg)
            methods[mode]={k:report[k] for k in keys}
            print(name,mode,{k:report[k] for k in keys[3:8]},flush=True)
        old_case=json.loads((previous/name/'support_aware/report.json').read_text())
        for key in keys[:8]:
            if methods['legacy_unrotated'][key]!=old_case[key]:raise AssertionError('Historical baseline did not reproduce: '+key)
        for key in keys[:3]:assert methods['legacy_unrotated'][key]==methods['source_heading'][key]
        cases.append(dict(id=name,historical_baseline_reproduced=True,source_coverage_and_tracking_ticks_preserved=True,methods=methods))
    result=dict(freeze=freeze,cases=cases)
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
