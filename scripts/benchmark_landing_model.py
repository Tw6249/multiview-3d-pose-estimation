"""Analytic model ablation, explicitly separate from visual or robot validation."""
import hashlib
import itertools
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motion_control.landing_planner import LandingPlanner,load_config


def run(output):
    output=Path(output)
    if output.exists():raise FileExistsError('Keep prior analytic evidence')
    output.mkdir(parents=True);cfg=load_config();states=list(itertools.product((.1,.4,.7),(-.1,0,.1),(-1.,0,1.),(-1.,0,1.)))
    freeze=dict(configuration=cfg,grid_vx_vy_Lx_Ly=states,scope='Synthetic centroidal states only. No cameras, human observations, full-body dynamics or hardware validation.',
        code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),ROOT/'motion_control/landing_planner.py')})
    (output/'freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8');cases=[];planner=LandingPlanner('analytic_model',cfg)
    for vx,vy,lx,ly in states:
        robot=dict(timestamp_ns=0,clock_domain='offline_simulation',target_model_sha256='analytic_model',
            com_world_m=[0,.05,.65],com_velocity_world_m_s=[vx,vy,0],angular_momentum_world_kg_m2_s=[lx,ly,0],mass_kg=33.,gravity_m_s2=9.81,
            loaded_contact_points_by_foot_world_m=dict(right=[[-.08,-.03,0],[.08,-.03,0],[.08,.03,0],[-.08,.03,0]],left=[]))
        event=dict(phase='observed_descending_swing',side='left',stance_side='right',source_timestamp_ns=0,observation_robot_ns=0,
            earliest_robot_ns=120_000_000,latest_robot_ns=320_000_000,nominal_robot_ns=220_000_000,
            nominal_landing_world_xy_m=[.15,.1],desired_world_velocity_m_s=[.3,0],target_heading_rad=0.)
        methods={}
        for mode in ('fixed','position','joint'):
            r=planner.solve(robot,event,0,mode);methods[mode]={k:v for k,v in r.items() if k!='attempts'}
        cases.append(dict(state_vx_vy_Lx_Ly=[vx,vy,lx,ly],methods=methods))
    result=dict(freeze=freeze,cases=cases,feasible_counts={m:sum(c['methods'][m]['status']=='model_feasible' for c in cases) for m in ('fixed','position','joint')})
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(result['feasible_counts'],flush=True);return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
