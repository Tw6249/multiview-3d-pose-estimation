from itertools import combinations
import numpy as np
import cv2
from scipy.optimize import least_squares
from .uncertainty import linearized_covariance, covariance_from_jacobian

JOINTS = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
          'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
          'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
          'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
         (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
         (0, 1), (0, 2), (1, 3), (2, 4)]


def rigid_transform(value):
    """Validate a finite proper SE(3) transform, independent of any detector."""
    transform = np.asarray(value, float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all() or
            not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8, rtol=0) or
            not np.allclose(transform[:3, :3].T@transform[:3, :3], np.eye(3), atol=1e-6, rtol=0) or
            not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-6, rtol=0)):
        raise ValueError('Expected a proper SE(3) transform')
    return transform


def arrays(cam):
    return (np.asarray(cam['K'], float), np.asarray(cam['R_world_to_camera'], float),
            np.asarray(cam['t_world_to_camera_m'], float), np.asarray(cam.get('distortion', [0]*5), float))


def project(xyz, cam):
    k, r, t, d = arrays(cam)
    p, _ = cv2.projectPoints(np.asarray(xyz, float).reshape(-1, 3), cv2.Rodrigues(r)[0], t, k, d)
    return p.reshape(-1, 2)


def center(cam):
    _, r, t, _ = arrays(cam)
    return -r.T @ t


def dlt(cams, xy, weights):
    rows = []
    for cam, pixel, w in zip(cams, xy, weights):
        k, r, t, d = arrays(cam)
        u, v = cv2.undistortPoints(np.asarray(pixel, float).reshape(1, 1, 2), k, d)[0, 0]
        p = np.column_stack((r, t))
        rows.extend([np.sqrt(w) * (u*p[2]-p[0]), np.sqrt(w) * (v*p[2]-p[1])])
    _, _, vt = np.linalg.svd(rows)
    if abs(vt[-1, 3]) < 1e-10:
        raise ValueError('Point at infinity')
    return vt[-1, :3] / vt[-1, 3]


class CameraGeometry:
    """Precompute camera matrices; vectorize pinhole projection and its Jacobian."""
    def __init__(self, cams):
        self.cams = cams
        parts = [arrays(c) for c in cams]
        self.k = np.array([p[0] for p in parts]);self.r = np.array([p[1] for p in parts])
        self.t = np.array([p[2] for p in parts]);self.distortion = [p[3] for p in parts]
        self.p = np.concatenate((self.r,self.t[:,:,None]),axis=2)
        self.centers = -np.einsum('nji,nj->ni',self.r,self.t)
        self.pinhole = all(not np.any(d) for d in self.distortion)

    def project(self, x):
        if not self.pinhole:
            return np.array([project(x,c)[0] for c in self.cams])
        v = np.einsum('nij,j->ni',self.r,x)+self.t
        return v[:,:2]/v[:,2:]*self.k[:,[0,1],[0,1]]+self.k[:,:2,2]

    def jacobian(self,x):
        if not self.pinhole:
            h=1e-5
            return np.stack([(self.project(x+h*a)-self.project(x-h*a))/(2*h) for a in np.eye(3)],axis=-1)
        v=np.einsum('nij,j->ni',self.r,x)+self.t
        j=(self.r[:,:2,:]*v[:,2,None,None]-v[:,:2,None]*self.r[:,2:3,:])/v[:,2,None,None]**2
        return j*self.k[:,[0,1],[0,1]][:,:,None]

    def dlt_rows(self,xy,scores):
        if self.pinhole:
            uv=(np.asarray(xy)-self.k[:,:2,2])/self.k[:,[0,1],[0,1]]
            return (uv[:,:,None]*self.p[:,2:3,:]-self.p[:,:2,:])*np.sqrt(scores)[:,None,None]
        uv=[]
        for i,pixel in enumerate(xy):
            if self.pinhole:
                uv.append((pixel-self.k[i,:2,2])/self.k[i,[0,1],[0,1]])
            else:
                uv.append(cv2.undistortPoints(np.asarray(pixel,float).reshape(1,1,2),self.k[i],self.distortion[i])[0,0])
        return (np.asarray(uv)[:,:,None]*self.p[:,2:3,:]-self.p[:,:2,:])*np.sqrt(scores)[:,None,None]


def refine_soft_l1(x,residual,jacobian,iterations=12):
    """Small CPU soft-L1 solve, monotone line search; None requests SciPy fallback.

    Residuals/Jacobians have exactly the same confidence weights and 3 px scale
    as the reference solver. No changed inlier thresholds or fabricated evidence.
    """
    x=np.array(x,float,copy=True)
    def cost(r):return float(np.sum(18*(np.sqrt(1+(r/3)**2)-1)))
    for _ in range(iterations):
        r=residual(x);j=jacobian(x)
        w=1/np.sqrt(1+(r/3)**2);h=j.T@((w**3)[:,None]*j);g=j.T@(w*r)
        if not np.isfinite(h).all():return None
        try:step=np.linalg.solve(h+np.eye(3)*1e-9,g)
        except np.linalg.LinAlgError:return None
        if np.linalg.norm(step)<1e-7:return x
        before=cost(r);accepted=False
        for scale in (1.,.5,.25,.125,.0625):
            candidate=x-scale*step
            after=cost(residual(candidate))
            if after<=before+1e-12:
                x=candidate;accepted=True;break
        if not accepted:return None
        if abs(before-after)<1e-10*max(1.,before):return x
    return None


def triangulate(cams, xy, scores, threshold_px=8.0, min_angle_deg=1.0, geometry=None, solver='scipy'):
    """Pair-enumeration RANSAC, weighted DLT, robust pixel reprojection refinement."""
    xy, scores = np.asarray(xy, float), np.asarray(scores, float)
    if solver not in ('scipy','newton'):raise ValueError('Unknown triangulation solver')
    if len(cams) < 2:
        return None
    geometry = geometry or CameraGeometry(cams)
    rows = geometry.dlt_rows(xy,scores)
    def solve_dlt(indices):
        _,_,vt=np.linalg.svd(rows[indices].reshape(-1,4))
        if abs(vt[-1,3])<1e-10:raise ValueError('Point at infinity')
        return vt[-1,:3]/vt[-1,3]
    def errors(x):
        return np.linalg.norm(geometry.project(x)-xy,axis=1)
    def positive(x, indices):
        return bool(np.all((geometry.r[indices]@x+geometry.t[indices])[:,2]>0))
    best = None
    for pair in combinations(range(len(cams)), 2):
        try:
            x = solve_dlt(list(pair))
        except (ValueError, np.linalg.LinAlgError):
            continue
        e = errors(x)
        ids = np.flatnonzero(e <= threshold_px)
        if len(ids) < 2 or not positive(x, ids):
            continue
        rank = (len(ids), float(scores[ids].sum()), -float(e[ids].mean()))
        if best is None or rank > best[0]:
            best = rank, ids, x
    if best is None:
        return None
    ids = best[1]
    fallback=False
    for _ in range(2):
        x = solve_dlt(ids)
        def residual(v):
            return ((geometry.project(v)[ids]-xy[ids])*np.sqrt(scores[ids,None])).ravel()
        def jacobian(v):
            return (geometry.jacobian(v)[ids]*np.sqrt(scores[ids,None,None])).reshape(-1,3)
        candidate=refine_soft_l1(x,residual,jacobian) if solver=='newton' else None
        if candidate is None:
            fallback=fallback or solver=='newton'
            x = least_squares(residual, x, jac=jacobian, loss='soft_l1', f_scale=3.0, max_nfev=40).x
        else:x=candidate
        new_ids = np.flatnonzero(errors(x) <= threshold_px)
        if len(new_ids) < 2:
            return None
        if np.array_equal(new_ids, ids):
            break
        ids = new_ids
    if not np.isfinite(x).all() or not positive(x, ids):
        return None
    rays = x-geometry.centers[ids]
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    angles = [np.degrees(np.arccos(np.clip(abs(np.dot(rays[a], rays[b])), 0, 1)))
              for a, b in combinations(range(len(rays)), 2)]
    angle = max(angles)
    if angle < min_angle_deg:
        return None
    uncertainty = covariance_from_jacobian(geometry.jacobian(x)[ids].reshape(-1,3))
    if uncertainty is None:
        return None
    return {'xyz_m': x.tolist(), 'inliers': ids.tolist(),
            'mean_reprojection_px': float(errors(x)[ids].mean()), 'ray_angle_deg': float(angle),
            'refinement_solver':solver,'solver_fallback':fallback,
            **uncertainty}
