"""Typed G1 motor command boundary. Simulation is the only implemented sink."""
import numpy as np
from .unitree import SDK_JOINT_NAMES
from .twist import KP,KD


class G1SimulationSink:
    def __init__(self,plant):
        self.plant=plant;self.sequence=-1;self.timestamp=-1

    def validate(self,packet,now_ns):
        if packet.get('hardware_enabled') is not False or packet.get('clock_domain')!='offline_simulation':
            raise ValueError('This sink accepts simulation commands only')
        if packet.get('target_model_sha256')!=self.plant.sha256 or tuple(packet['joint_names'])!=SDK_JOINT_NAMES:
            raise ValueError('Model/joint identity mismatch')
        ts=packet['timestamp_ns'];expiry=packet['valid_until_ns'];seq=packet['sequence']
        if any(type(x) is not int for x in (ts,expiry,seq,now_ns)):
            raise ValueError('Integer sequence and nanosecond clocks required')
        if not ts<=now_ns<expiry or not 0<expiry-ts<=60_000_000:
            raise ValueError('Future/expired motor command')
        if seq<=self.sequence or ts<=self.timestamp:raise ValueError('Replayed motor command')
        q=np.asarray(packet['position_rad'],float)
        if q.shape!=(29,) or not np.isfinite(q).all() or np.any(q<self.plant.limits[:,0]) or np.any(q>self.plant.limits[:,1]):
            raise ValueError('Invalid joint target')
        for key,expected in [('kp',KP),('kd',KD),('effort_limit_nm',self.plant.effort),
                             ('velocity_rad_s',np.zeros(29)),('feedforward_torque_nm',np.zeros(29))]:
            if not np.array_equal(np.asarray(packet[key]),expected):raise ValueError('Command profile mismatch: '+key)
        self.sequence=seq;self.timestamp=ts
        return q
