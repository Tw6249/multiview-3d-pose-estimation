"""Reproducible split-isolated benchmark; RGB accuracy is a separate acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pose_sim.benchmark_data import generate
from pose_sim.pipeline import read_jsonl,write_jsonl,reconstruct_record
from pose_sim.human_state import HumanStateEstimator
from pose_sim.human_motion import HumanMotionEstimator
from pose_sim.multiview_temporal import MultiViewTemporal
from pose_sim.motion_evaluation import evaluate_motion,fit_uncertainty,apply_uncertainty,statistics,compare_methods


def implementation_hash():
    h=hashlib.sha256()
    for name in ('geometry.py','pipeline.py','uncertainty.py','landmarks.py','human_state.py','human_motion.py','body_shape.py','multiview_temporal.py','auxiliary.py','association_history.py','observation_consistency.py','reacquisition.py'):
        h.update((ROOT/'pose_sim'/name).read_bytes())
    h.update((ROOT/'config/vision_motion.json').read_bytes())
    return h.hexdigest()


def run(dataset,output,split,methods,calibration_path=None,save_states=False):
    dataset=Path(dataset);output=Path(output)
    if output.exists():raise FileExistsError('Preserve benchmark outputs; use a new run directory')
    output.mkdir(parents=True)
    manifest=json.loads((dataset/'manifest.json').read_text())
    signature=hashlib.sha256((implementation_hash()+json.dumps(manifest['configuration'],sort_keys=True)).encode()).hexdigest()
    cal=json.loads(Path(calibration_path).read_text()) if calibration_path else None
    report=dict(split=split,domain=manifest['domain'],configuration_sha256=signature,sequences=[],
                dataset_manifest_sha256=hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest())
    for entry in manifest['sequences']:
        if entry['split']!=split:continue
        folder=dataset/entry['id'];dest=output/entry['id']
        if save_states:dest.mkdir(parents=True)
        for file,sha in entry['sha256'].items():
            if hashlib.sha256((folder/file).read_bytes()).hexdigest()!=sha:raise ValueError('Benchmark input was modified')
        cameras=json.loads((folder/'cameras.json').read_text());cfg=json.loads((folder/'config.json').read_text())
        observations=list(read_jsonl(folder/'detections.jsonl'))
        # Reference labels are opened only after all perception for this sequence.
        result={};timings={}
        for method in methods:
            settings=json.loads((ROOT/'config/vision_motion.json').read_text())
            detail=method not in ('legacy_joint','no_auxiliary')
            history=method=='with_history' or (settings.get('association_history',False) and method not in ('legacy_joint','no_history'))
            state=HumanStateEstimator({'ground_height_m':0.}) if method=='baseline' else HumanMotionEstimator({**settings['state'],'ground_height_m':0.,'auxiliary_consistency':detail,'state_reacquisition':method in ('recovery','candidate')})
            fusion=MultiViewTemporal(cameras,cfg,association=method!='no_association',structured=method=='with_structure',auxiliary=detail,history=history,observation_consistency=method in ('consistency','candidate'))
            previous=None;rows=[];elapsed=[];corrections=0
            for detection in observations:
                start=time.perf_counter()
                if method=='baseline':
                    pose=reconstruct_record(detection,cameras['cameras'],cfg);assigned=detection
                    pose['calibration_sha256']=entry['sha256']['cameras.json']
                else:pose,assigned=fusion.update(detection,previous)
                previous=state.update(pose,cameras['cameras'],assigned)
                if method!='baseline':apply_uncertainty(previous,cal,manifest['domain'],signature)
                elapsed.append((time.perf_counter()-start)*1000)
                corrections+=len(pose.get('association',{}).get('corrections',[]));rows.append(previous)
            result[method]=rows;timings[method]=dict(processing_ms=statistics(elapsed),corrected_view_groups=corrections)
            if save_states:write_jsonl(dest/(method+'.jsonl'),rows)
        truth=list(read_jsonl(folder/'ground_truth.jsonl'))
        metrics={m:{**evaluate_motion(rows,truth),**timings[m]} for m,rows in result.items()}
        report['sequences'].append(dict(id=entry['id'],subject=entry['subject'],action=entry['action'],methods=metrics,
                                       paired=compare_methods(result,truth)))
        print(entry['id'],{m:round(v['position_error_mm']['mean'],2) for m,v in metrics.items()},flush=True)
    if split=='calibration' and 'joint' in methods:
        calibration=fit_uncertainty([s['methods']['joint']['normalized_error_max'] for s in report['sequences']],manifest['domain'],signature)
        calibration['subjects']=sorted({s['subject'] for s in report['sequences']})
        (output/'uncertainty_calibration.json').write_text(json.dumps(calibration,indent=2),encoding='utf-8')
    report['aggregate']={m:{'mean_sequence_position_error_mm':sum(s['methods'][m]['position_error_mm']['mean'] for s in report['sequences'])/len(report['sequences']),
        'mean_coverage':sum(s['methods'][m]['coverage'] for s in report['sequences'])/len(report['sequences']),
        'mean_processing_ms':sum(s['methods'][m]['processing_ms']['mean'] for s in report['sequences'])/len(report['sequences'])} for m in methods}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report['aggregate'],indent=2));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--generate',action='store_true')
    p.add_argument('--config',type=Path,default=ROOT/'config/vision_benchmark.json');p.add_argument('--output',type=Path)
    p.add_argument('--split',choices=('tune','calibration','test'),default='tune');p.add_argument('--methods',nargs='+',default=['baseline','joint'],choices=('baseline','joint','no_association','with_structure','legacy_joint','no_auxiliary','no_history','with_history','consistency','recovery','candidate'))
    p.add_argument('--calibration',type=Path);p.add_argument('--save-states',action='store_true');args=p.parse_args()
    if args.generate:generate(args.dataset,json.loads(args.config.read_text()))
    if args.output:run(args.dataset,args.output,args.split,args.methods,args.calibration,args.save_states)
