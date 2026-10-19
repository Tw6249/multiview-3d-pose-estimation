"""Diagnostic source-phase-constrained centroidal landing QP; no motor output.

Robot COM (m,m/s), horizontal angular momentum (kg*m^2/s), loaded stance points
and a source-derived landing location/time window -> one-step plan or unknown.
Z-up world, robot clock integer ns; source timestamp retained separately.
Constant COM height/vertical force, point contact after touchdown, linear
momentum segments and terminal DCM relation are model assumptions. Joint torque,
whole-body reach, swing clearance and collision feasibility are NOT certified.
"""
import copy
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import qr
from .root_tracking import planar_rotation


def load_config():return json.loads((Path(__file__).resolve().parents[1]/'config/g1_landing_planner.json').read_text())


def propagate(c,v,cmp,omega,seconds):
    """Exact constant-CMP inverted-pendulum flow; accepts affine coefficient arrays."""
    a=np.cosh(omega*seconds);b=np.sinh(omega*seconds)
    return a*c+b/omega*v+(1-a)*cmp,omega*b*c+a*v-omega*b*cmp


class LandingPlanner:
    def __init__(self,model_sha256,config=None):
        self.cfg=copy.deepcopy(config or load_config());self.model_sha256=model_sha256
        if self.cfg.get('hardware_enabled') is not False or self.cfg.get('target')!='unitree_g1_29dof':raise ValueError('Diagnostic G1 profile required')
        for key in ('after_landing_seconds','source_landing_corridor_m','friction_coefficient','maximum_angular_momentum_kg_m2_s',
                    'maximum_angular_momentum_rate_nm','minimum_com_height_m','maximum_com_height_m','velocity_cost_scale_m_s',
                    'position_cost_scale_m','landing_cost_scale_m','timing_cost_scale_s','maximum_robot_age_s','maximum_source_age_s',
                    'minimum_landing_time_s','maximum_landing_time_s','momentum_cost_weight','constraint_tolerance'):
            if not np.isfinite(self.cfg[key]) or self.cfg[key]<=0:raise ValueError('Invalid planner bound: '+key)
        if type(self.cfg['landing_time_samples']) is not int or not 1<=self.cfg['landing_time_samples']<=15:raise ValueError('Bound the landing-time enumeration')
        if self.cfg['minimum_landing_time_s']>=self.cfg['maximum_landing_time_s'] or self.cfg['minimum_com_height_m']>=self.cfg['maximum_com_height_m']:raise ValueError('Invalid planning envelope')
        if type(self.cfg['solver_max_iterations']) is not int or not 1<=self.cfg['solver_max_iterations']<=1000:raise ValueError('Invalid solver iteration budget')

    def solve(self,robot,event,now_ns,mode='joint'):
        start=time.perf_counter();cfg=self.cfg
        def reject(reason):return dict(status='unknown',reason=reason,plan=None,hardware_enabled=False,wall_ms=(time.perf_counter()-start)*1000)
        if mode not in ('fixed','position','joint'):raise ValueError('Unknown centroidal ablation')
        if type(now_ns) is not int or robot.get('clock_domain')!='offline_simulation' or robot.get('target_model_sha256')!=self.model_sha256:raise ValueError('Robot model/clock mismatch')
        stamp=robot.get('timestamp_ns')
        if type(stamp) is not int or not 0<=now_ns-stamp<=round(cfg['maximum_robot_age_s']*1e9):return reject('Stale or future robot feedback')
        if event is None:return reject('Source landing phase unknown')
        if event.get('phase')!='observed_descending_swing' or event.get('side') not in ('left','right') or event.get('stance_side') not in ('left','right') or event['side']==event['stance_side']:return reject('Invalid source phase or side identity')
        for key in ('source_timestamp_ns','observation_robot_ns','earliest_robot_ns','latest_robot_ns','nominal_robot_ns'):
            if type(event.get(key)) is not int:return reject('Integer mapped source clocks required')
        if not 0<=now_ns-event['observation_robot_ns']<=round(cfg['maximum_source_age_s']*1e9):return reject('Source observation is stale or from the future')
        lo=max(cfg['minimum_landing_time_s'],(event['earliest_robot_ns']-now_ns)/1e9)
        hi=min(cfg['maximum_landing_time_s'],(event['latest_robot_ns']-now_ns)/1e9)
        if lo>hi:return reject('Observed landing window expired or outside horizon')
        nominal=np.clip((event['nominal_robot_ns']-now_ns)/1e9,lo,hi)
        c=np.asarray(robot['com_world_m'],float);v=np.asarray(robot['com_velocity_world_m_s'],float)
        momentum=np.asarray(robot['angular_momentum_world_kg_m2_s'],float)
        points=np.asarray(robot['loaded_contact_points_by_foot_world_m'][event['stance_side']],float).reshape(-1,3)
        landing=np.asarray(event['nominal_landing_world_xy_m'],float);desired=np.asarray(event['desired_world_velocity_m_s'],float)
        mass=robot['mass_kg'];g=robot['gravity_m_s2'];heading=event['target_heading_rad']
        if c.shape!=(3,) or v.shape!=(3,) or momentum.shape!=(3,) or landing.shape!=(2,) or desired.shape!=(2,) or not np.isfinite(np.r_[c,v,momentum,points.ravel(),landing,desired,mass,g,heading]).all() or min(mass,g)<=0:raise ValueError('Invalid centroidal state')
        if len(points)==0:return reject('Source stance lacks current robot support')
        height=c[2]-np.median(points[:,2])
        if not cfg['minimum_com_height_m']<=height<=cfg['maximum_com_height_m']:return reject('COM height outside constant-height model')
        R=planar_rotation(heading);J=np.array([[0.,1.],[-1.,0.]])
        # Shift the world origin for conditioning. CMP offset = J * Ldot / (m*g).
        origin=points[:,:2].mean(axis=0);c0=c[:2]-origin;target=landing-origin;ell0=J@momentum[:2]/(mass*g)
        omega=np.sqrt(g/height);after=cfg['after_landing_seconds'];rate=cfg['maximum_angular_momentum_rate_nm']/(mass*g)
        maxell=cfg['maximum_angular_momentum_kg_m2_s']/(mass*g);corridor=cfg['source_landing_corridor_m']
        n=len(points);d=n+4
        P=np.zeros((2,d+1));P[:,1:1+n]=(points[:,:2]-origin).T
        F=np.zeros_like(P);F[:,0]=target;F[:,1+n:3+n]=R*corridor
        K=np.zeros_like(P);K[:,3+n:5+n]=R*rate
        C=np.zeros_like(P);C[:,0]=c0;V=np.zeros_like(P);V[:,0]=v[:2]
        times=np.unique(np.r_[lo,nominal,hi,np.linspace(lo,hi,cfg['landing_time_samples'])]) if mode=='joint' else [nominal]
        attempts=[]
        for duration in times:
            K2=-duration/after*K;K2[:,0]-=ell0/after
            C1,V1=propagate(C,V,P+K,omega,duration);C2,V2=propagate(C1,V1,F+K2,omega,after)
            capture=C2+V2/omega-F;capture[:,0]-=desired/omega
            equality=np.r_[np.r_[-1.,np.ones(n),np.zeros(4)][None],capture]
            momentum_end=duration*K;momentum_end[:,0]+=ell0
            constrained=[R.T@K2/rate,R.T@momentum_end/maxell]
            # Sampled friction constraints, not a continuous-time force certificate.
            friction=[];directions=np.array([[1,1],[1,-1],[-1,1],[-1,-1]])@R.T
            for t in np.linspace(0,duration,6):
                ct,_=propagate(C,V,P+K,omega,t);friction.append(directions@(omega**2*(ct-P-K))/(cfg['friction_coefficient']*g))
            for t in np.linspace(0,after,6):
                ct,_=propagate(C1,V1,F+K2,omega,t);friction.append(directions@(omega**2*(ct-F-K2))/(cfg['friction_coefficient']*g))
            inequalities=np.concatenate([a for m in constrained for a in (m,-m)]+friction)
            initial_ok=np.max(np.abs(R.T@ell0))<=maxell+1e-12
            velocity=V2.copy();velocity[:,0]-=desired
            shift=F.copy();shift[:,0]-=target
            residual=np.r_[velocity/cfg['velocity_cost_scale_m_s'],(C2-F)/cfg['position_cost_scale_m'],
                shift/cfg['landing_cost_scale_m'],np.sqrt(cfg['momentum_cost_weight'])*K/rate]
            A=residual[:,1:];b=residual[:,0]
            def cost(x):r=A@x+b;return float(r@r),2*A.T@r
            bounds=[(0,1)]*n+[(-1,1)]*4
            if mode=='fixed':bounds[n:n+2]=[(0,0)]*2
            if mode in ('fixed','position'):bounds[-2:]=[(0,0)]*2
            x0=np.r_[np.full(n,1/n),np.zeros(4)]
            free=[i for i,bound in enumerate(bounds) if bound[0]!=bound[1]]
            _,rr,pivot=qr(equality[:,1:][:,free].T,pivoting=True)
            rank=np.linalg.matrix_rank(rr,tol=1e-10);independent=equality[pivot[:rank]]
            constraints=[dict(type='ineq',fun=lambda x:1-inequalities[:,0]-inequalities[:,1:]@x,jac=lambda x:-inequalities[:,1:])]
            if rank:constraints.append(dict(type='eq',fun=lambda x:independent[:,1:]@x+independent[:,0],jac=lambda x:independent[:,1:]))
            fit=minimize(cost,x0,jac=True,method='SLSQP',bounds=bounds,constraints=constraints,
                options=dict(maxiter=cfg['solver_max_iterations'],ftol=1e-9))
            x=np.r_[1.,fit.x];eq=float(np.max(np.abs(equality@x)));ineq=float(np.max(inequalities@x-1))
            good=bool(initial_ok and fit.success and np.isfinite(fit.x).all() and eq<=cfg['constraint_tolerance'] and ineq<=cfg['constraint_tolerance'])
            attempts.append(dict(feasible=good,landing_after_s=float(duration),cost=float(fit.fun+((duration-nominal)/cfg['timing_cost_scale_s'])**2),
                equality_residual=eq,inequality_violation=max(0.,ineq),solver_message=str(fit.message),
                landing_world_xy_m=(F@x+origin).tolist(),cop_before_world_xy_m=(P@x+origin).tolist(),
                angular_momentum_rate_before_world_nm=(J.T@(K@x)*(mass*g)).tolist(),
                angular_momentum_rate_after_world_nm=(J.T@(K2@x)*(mass*g)).tolist(),
                angular_momentum_at_landing_world_kg_m2_s=(J.T@(momentum_end@x)*(mass*g)).tolist(),
                terminal_com_world_xy_m=(C2@x+origin).tolist(),terminal_velocity_world_m_s=(V2@x).tolist()))
        feasible=[r for r in attempts if r['feasible']];winner=min(feasible,key=lambda r:r['cost']) if feasible else None
        return dict(status='model_feasible' if winner else 'no_feasible_candidate',plan=winner,attempts=attempts,
            source_timestamp_ns=event['source_timestamp_ns'],side=event['side'],mode=mode,hardware_enabled=False,
            assumptions='Constant-height point-contact model, zero terminal horizontal angular momentum, source landing window. Not whole-body executable or guaranteed stable.',
            wall_ms=(time.perf_counter()-start)*1000)
