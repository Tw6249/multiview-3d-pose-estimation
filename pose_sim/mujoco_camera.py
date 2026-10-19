"""Convert calibrated OpenCV cameras to MuJoCo OpenGL camera attributes."""
import numpy as np


def camera_attributes(camera):
    r=np.asarray(camera['R_world_to_camera'],float);t=np.asarray(camera['t_world_to_camera_m'],float)
    k=np.asarray(camera['K'],float);w,h=camera['width'],camera['height']
    if np.any(np.asarray(camera.get('distortion',[0]*5))!=0) or k[0,1]!=0:
        raise ValueError('Render adapter requires undistorted zero-skew intrinsics')
    axes=r.T@np.diag([1,-1,-1])
    # MuJoCo sensor shifts use x opposite to OpenCV principal-point shifts.
    return {'pos':' '.join(map(str,-r.T@t)),'xyaxes':' '.join(map(str,np.r_[axes[:,0],axes[:,1]])),
            'resolution':f'{w} {h}','sensorsize':f'{w} {h}',
            'focalpixel':f'{k[0,0]} {k[1,1]}','principalpixel':f'{w/2-k[0,2]} {k[1,2]-h/2}'}
