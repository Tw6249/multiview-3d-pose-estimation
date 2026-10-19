"""Freeze block-wise reference-error scales, then inspect heldout coverage."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from pose_sim.empirical_bounds import fit_bounds,block_maxima
from pose_sim.motion_evaluation import statistics


def run(root):
    root=Path(root);destination=root/'uncertainty_audit.json'
    if destination.exists():raise FileExistsError(destination)
    config=json.loads((root/'calibration/freeze.json').read_text());held=json.loads((root/'heldout/freeze.json').read_text())
    if config['implementation_sha256']!=held['implementation_sha256']:raise ValueError('Calibration/assessment implementations differ')
    result={}
    for quantity,error_key,std_key,floor in [('position','error_m','sigma_m',.005),('velocity','velocity_error_m_s','velocity_sigma_m_s',.05)]:
        scores=[]
        for path in sorted((root/'calibration').glob('*baseline_scores.npz')):
            d=np.load(path);scores+=block_maxima(d[error_key],d[std_key],d['timestamp_ns'],d['current'],floor)
        cal=fit_bounds(scores,'mpi3dhp_four_view_body12_'+quantity,config['implementation_sha256'],std_floor=floor)
        # Save the fit before opening any heldout score arrays.
        (root/(quantity+'_calibration.json')).write_text(json.dumps(cal,indent=2),encoding='utf-8')
        clips=[]
        for path in sorted((root/'heldout').glob('*baseline_scores.npz')):
            d=np.load(path);valid=d['current']&np.isfinite(d[error_key])&np.isfinite(d[std_key]);e=d[error_key][valid];s=np.maximum(floor,d[std_key][valid]);r=s*cal['multiplier']
            blocks=block_maxima(d[error_key],d[std_key],d['timestamp_ns'],d['current'],floor)
            clips.append(dict(id=path.stem,targets=len(valid),current_scored=int(valid.sum()),
                current_scored_fraction=float(valid.mean()),nominal_three_std_reference_coverage=float(np.mean(e<=3*s)),
                calibrated_reference_coverage=float(np.mean(e<=r)),all_target_within_radius=float(np.sum(e<=r)/len(valid)),
                block_reference_coverage=float(np.mean(np.array(blocks)<=cal['multiplier'])),blocks=len(blocks),radius=statistics(r.tolist()),
                risk_at_radius_threshold=[dict(maximum_radius=threshold,accepted_targets=int((r<=threshold).sum()),
                    all_target_accepted_fraction=float((r<=threshold).sum()/len(valid)),
                    error_exceeds_threshold_fraction=float(np.mean(e[r<=threshold]>threshold)) if (r<=threshold).any() else None)
                    for threshold in ([.05,.1,.2] if quantity=='position' else [.2,.5,1.])]))
        result[quantity]=dict(calibration=cal,clips=clips)
    result['scope']='Descriptive reference-error calibration only. Adjacent blocks dependent, known dataset/subjects; no true contact/rotation/force or hardware coverage guarantee. Default control gate unchanged.'
    destination.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)
