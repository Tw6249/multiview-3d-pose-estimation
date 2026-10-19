"""Post-selection model sensitivity; never retune or choose a new candidate."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motion_control.humanoid_replay import replay
from motion_control.recovery_rollout import RecoveryRollout,snapshot,branch,boundary_reference

CONDITIONS=(('mass_90_percent',.9,1.),('mass_110_percent',1.1,1.),
            ('friction_50_percent',1.,.5),('friction_125_percent',1.,1.25))


def altered_model(plant,policy,mass_factor,friction_factor):
    """Independent nominal geometry with changed mass/inertia or geom friction."""
    p,_=branch(plant,policy,snapshot(plant));p.model=copy.copy(plant.model)
    p.model.body_mass[:]*=mass_factor;p.model.body_inertia[:]*=mass_factor
    p.model.geom_friction[:,0]*=friction_factor
    p.data=p.mj.MjData(p.model);p.mj.mj_setConst(p.model,p.data)
    p.mj.mj_setState(p.model,p.data,snapshot(plant),p.mj.mjtState.mjSTATE_INTEGRATION)
    p.sha256=hashlib.sha256((plant.sha256+repr((mass_factor,friction_factor))).encode()).hexdigest()
    return p


def run(validation,output):
    validation=Path(validation);output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve sensitivity evidence')
    output.mkdir(parents=True);base=json.loads((validation/'report.json').read_text())
    for name,sha in base['freeze']['code_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('Controller changed since frozen validation: '+name)
    freeze=dict(conditions=CONDITIONS,validation_sha256=hashlib.sha256((validation/'report.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),selection='Keep previously selected schedule; no new search or parameter update',
        scope='Independent copies of target robot simulation model. No source truth, hardware or calibrated uncertainty.')
    (output/'freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8');cases=[]
    for name,source_sha in base['freeze']['source_sha256'].items():
        source=Path(name);case_id=source.parent.name if source.name in ('baseline','integrated_displacement') else source.name
        old=json.loads((validation/case_id/'rollout/report.json').read_text());diag=old['recovery_rollout']
        if not diag['accepted']:continue
        if hashlib.sha256((source/'humanoid_reference.jsonl').read_bytes()).hexdigest()!=source_sha:raise ValueError('Source changed')
        parameters=diag['candidates'][diag['selected']]['parameters'];results=[]
        def evaluate(planner,plant,policy,applied,timestamp_ns,clock_domain='offline_simulation'):
            entry=boundary_reference(plant,applied,parameters[3]);saved=snapshot(plant)
            nominal=planner.verify(plant,policy,saved,applied,entry,parameters)
            expected=next(x for x in diag['cpu_verifications'] if x['id']==diag['selected'])
            for key in nominal:
                if nominal[key]!=expected[key]:raise AssertionError('Frozen nominal replay did not reproduce: '+key)
            for label,mass,friction in CONDITIONS:
                p=altered_model(plant,policy,mass,friction)
                result=planner.verify(p,policy,saved,applied,entry,parameters)
                results.append(dict(condition=label,mass_scale=mass,friction_scale=friction,**result))
            plan=dict(timestamp_ns=timestamp_ns,model_sha256=plant.sha256,entry=entry,parameters=parameters)
            return plan,dict(accepted=True,selected=diag['selected'],nominal_reproduced_exactly=True,model_sensitivity=results,
                online_eligible=False,hardware_enabled=False,source_future_used=False)
        with patch.object(RecoveryRollout,'plan',evaluate):
            report=replay(source,video=False,save_logs=False,output_name=str(output/case_id),
                transition_mode='rollout',recovery_config=base['freeze']['configuration'])
        if report['standing_stop_audit']!=old['standing_stop_audit']:raise AssertionError('Audit changed the nominal controller')
        cases.append(dict(id=case_id,selected=diag['selected'],nominal_reproduced_exactly=True,conditions=results))
        print(case_id,[(r['condition'],r['feasible']) for r in results],flush=True)
    result=dict(freeze=freeze,cases=cases)
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--validation',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.validation,args.output)
