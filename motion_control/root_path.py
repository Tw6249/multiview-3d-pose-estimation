"""Size-retargeted pelvis displacement, independent of the world origin.

Input: accepted source pelvis XY in world metres, positive target/source length
ratio, strictly increasing exposure timestamp in integer ns. Output: target XY
metres, with source world axes and zero at the first accepted pelvis. Each new
displacement uses the current ratio; recalibrating body size cannot move a
stationary person. No interpolation, contacts, orientation or hardware commands.
The caller gates missing observations; invalid input leaves history unchanged.
"""
import numpy as np


class ScaledRootPath:
    def __init__(self):
        self.previous=None
        self.timestamp_ns=None
        self.position=np.zeros(2)

    def update(self,source_xy_m,scale_to_target,timestamp_ns):
        source=np.asarray(source_xy_m,float)
        if source.shape!=(2,) or not np.isfinite(source).all():
            raise ValueError('Invalid source pelvis position')
        if not np.isfinite(scale_to_target) or scale_to_target<=0:
            raise ValueError('Invalid target/source length ratio')
        if type(timestamp_ns) is not int or (self.timestamp_ns is not None and timestamp_ns<=self.timestamp_ns):
            raise ValueError('Nonmonotonic source pelvis clock')
        position=self.position.copy()
        if self.previous is not None:
            position+=(source-self.previous)*scale_to_target
        if not np.isfinite(position).all():raise ValueError('Nonfinite retargeted pelvis path')
        self.position=position
        self.previous=source.copy()
        self.timestamp_ns=timestamp_ns
        return position.copy()
