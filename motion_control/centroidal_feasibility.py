"""Instantaneous flat-ground braking-wrench feasibility, not a controller.

COM and contact XYZ m in one Z-up world, desired acceleration m/s^2, mass kg,
friction coefficient unitless -> feasible/infeasible/unknown and contact N.
Assumes zero angular-momentum rate, fixed contacts, unilateral normal forces and
an inner friction pyramid. Omits joint torque, self-collision, contact changes and
the future COM trajectory. A feasible instant does NOT certify a safe stop.
"""
import numpy as np
from scipy.optimize import linprog


def braking_wrench(com,contacts,mass,acceleration,friction=.6,gravity=9.81):
    com=np.asarray(com,float);points=np.asarray(contacts,float).reshape(-1,3);acc=np.asarray(acceleration,float)
    if com.shape!=(3,) or acc.shape!=(3,) or not np.isfinite(np.r_[com,points.ravel(),acc,mass,friction,gravity]).all() or min(mass,friction,gravity)<=0:
        raise ValueError('Invalid centroidal feasibility input')
    if not len(points):return dict(status='unknown',reason='No loaded contacts',forces_world_n=None)
    aeq=np.zeros((6,3*len(points)));inequality=[]
    for i,r in enumerate(points-com):
        col=slice(3*i,3*i+3);aeq[:3,col]=np.eye(3)
        aeq[3:,col]=[[0,-r[2],r[1]],[r[2],0,-r[0]],[-r[1],r[0],0]]
        for axis in (0,1):
            for sign in (-1,1):
                row=np.zeros(3*len(points));row[3*i+axis]=sign;row[3*i+2]=-friction/np.sqrt(2);inequality.append(row)
    rhs=np.r_[mass*(acc+[0,0,gravity]),np.zeros(3)]
    bounds=[b for _ in points for b in ((None,None),(None,None),(0.,2*mass*gravity))]
    fit=linprog(np.zeros(3*len(points)),A_ub=inequality,b_ub=np.zeros(len(inequality)),A_eq=aeq,b_eq=rhs,bounds=bounds,method='highs')
    result=dict(status='infeasible' if fit.status==2 else 'unknown',solver_status=int(fit.status),forces_world_n=None,
                desired_acceleration_world_m_s2=acc.tolist(),friction_coefficient=friction,
                scope='Instantaneous fixed-contact, zero-angular-momentum-rate model; not whole-body or stopping feasibility')
    if fit.success:
        residual=float(np.max(np.abs(aeq@fit.x-rhs)));ineq=float(np.max(np.asarray(inequality)@fit.x))
        if residual<1e-5 and ineq<1e-6:
            result.update(status='feasible',forces_world_n=fit.x.reshape(-1,3).tolist(),maximum_equilibrium_residual=residual)
    return result
