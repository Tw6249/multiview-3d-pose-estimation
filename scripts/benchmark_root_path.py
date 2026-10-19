"""Audit the world-origin scaling fix on existing source-derived references.

Legacy references stored pelvis_world_xy * current_scale. Recover that same
estimated pelvis and integrate its scaled displacement. No images, joints or
labels are changed. Both simulations are scored against the corrected common
path; this is an engineering regression, not an additional independent test set.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from pose_sim.pipeline import read_jsonl,write_jsonl
from motion_control.root_path import ScaledRootPath
from motion_control.humanoid_replay import replay


def run(sources,output):
    if output.exists():raise FileExistsError('Preserve root-path audit')
    output.mkdir(parents=True)
    freeze=dict(source_files={str(p):hashlib.sha256((p/'humanoid_reference.jsonl').read_bytes()).hexdigest() for p in sources},
        implementation_sha256=hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in sorted((ROOT/'motion_control').glob('*.py')))).hexdigest(),
        candidate='Integrate scaled source pelvis displacement; no parameter fitting',
        common_evaluation='Corrected source-derived path and identical original joint reference',hardware_enabled=False)
    (output/'freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8');cases=[]
    for source in sources:
        case=output/source.name;case.mkdir();path=ScaledRootPath();legacy=[];corrected=[]
        for original in read_jsonl(source/'humanoid_reference.jsonl'):
            if 'root_path_frame' in original or 'offline_trajectory' in original:
                raise ValueError('Only unregularized legacy absolute-scaled reference is supported')
            base=copy.deepcopy(original);candidate=copy.deepcopy(original)
            if original['accepted']:
                source_xy=np.asarray(original['root_position_m'][:2])/original['scale_to_target']
                xy=path.update(source_xy,original['scale_to_target'],original['timestamp_ns'])
                target=[*xy,original['root_position_m'][2]]
                for row in (base,candidate):
                    row['source_root_position_m']=target
                    row['source_root_heading_rad']=original['root_heading_rad']
                candidate['root_position_m']=target
                candidate['root_path_frame']='source_world_xy_anchored_at_first_accepted_pelvis; target_height_z'
            legacy.append(base);corrected.append(candidate)
        results={}
        for method,rows in (('legacy_absolute_scaling',legacy),('integrated_displacement',corrected)):
            folder=case/method;folder.mkdir();write_jsonl(folder/'humanoid_reference.jsonl',rows)
            (folder/'retargeting.json').write_text(json.dumps(dict(source=str(source),source_sha256=freeze['source_files'][str(source)],
                operation=method,ground_truth_used=False,evaluation='Common corrected source path'),indent=2),encoding='utf-8')
            results[method]=replay(folder,video=False,output_name='simulation',save_logs=False)
        cases.append(dict(id=source.name,methods=results))
        print(source.name,{m:{k:r[k] for k in ('passed','balance_passed','source_reference_leg_rmse_rad','root_path_error_p95_m')} for m,r in results.items()},flush=True)
    report=dict(freeze=freeze,cases=cases,scope='Coordinate correctness and simulation regression; source estimate is not independent motion truth.')
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sources',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.sources,a.output)
