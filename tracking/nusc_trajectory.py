"""
object's trajectory. A trajectory is a collection of information for each frame(nusc_object.py)
Two core functions: state predict and state update.
The 'state' here is generalized, including attribute info, motion info, geometric info, score info, etc.

In general, Poly-MOT combines count-based and confidence-based strategy to manage trajectory lifecycle. 
Specifically, we use the count-based strategy to initialize and unregister trajectories, while using the 
score-based strategy to penalize mismatched trajectories
"""

import pdb
import numpy as np
from .nusc_life_manage import LifeManagement
from .nusc_score_manage import ScoreManagement
from .nusc_geometry_manage import GeometryManagement
from motion_module.nusc_object import FrameObject
from motion_module.kalman_filter import LinearKalmanFilter, ExtendKalmanFilter


class Trajectory:
    def __init__(self, timestamp: int, config: dict, track_id: int, det_infos: dict):
        # init basic infos
        self.timestamp = timestamp
        self.cfg, self.tracking_id, self.class_label = config, track_id, det_infos['np_array'][-1]
        # manage tracklet's attribute
        self.life_management = LifeManagement(timestamp, config, self.class_label)
        # manage tracklet's score, predict/update/punish trackelet
        self.score_management = ScoreManagement(timestamp, config, self.class_label, det_infos)
        # manage for tracklet's motion/geometric infos
        KF_type = self.cfg['motion_model']['filter'][self.class_label]
        assert KF_type in ['LinearKalmanFilter', 'ExtendKalmanFilter'], "must use specific kalman filter"
        self.motion_management = globals()[KF_type](timestamp, config, track_id, det_infos)

        # decouple size and z-position (constant) from motion management, reduce computational overhead
        if self.cfg['geometry_model']['use'][self.class_label]:
            self.geometry_management = GeometryManagement(timestamp, det_infos, self.motion_management[timestamp], self.cfg['geometry_model'])

        # DE-FastPoly: store last measured radial velocity for association
        self.last_vr = det_infos.get('radial_vel', 0.0) if det_infos.get('has_valid_vr', False) else None
        # DE-FastPoly: RCS state (valid only when there are in-box radar points)
        init_rcs = det_infos.get('radar_rcs', None)
        init_npts = int(det_infos.get('vr_n_pts', 0)) if det_infos.get('vr_n_pts', None) is not None else 0
        if init_rcs is not None and np.isfinite(float(init_rcs)) and init_npts > 0:
            self.last_rcs = float(init_rcs)
            self.last_rcs_npts = init_npts
        else:
            self.last_rcs = None
            self.last_rcs_npts = 0
            
    
    def state_predict(self, timestamp: int) -> None:
        """
        predict trajectory's state
        :param timestamp: current frame id
        """
        self.timestamp = timestamp
        self.life_management.predict(timestamp)
        self.motion_management.predict(timestamp)
        if self.cfg['geometry_model']['use'][self.class_label]:
            self.geometry_management.predict(timestamp, self.motion_management[timestamp])
        self.score_management.predict(timestamp, self.motion_management[timestamp])
    
    def state_update(self, timestamp: int, det: dict = None) -> None:
        """
        update trajectory's state
        :param timestamp: current frame id
        :param det: dict, detection infos under different data format
            {
                'nusc_box': NuscBox,
                'np_array': np.array,
                'has_velo': bool, whether the detetor has velocity info
            }
        """
        self.timestamp = timestamp
        self.motion_management.update(timestamp, det)
        if self.cfg['geometry_model']['use'][self.class_label]:
            self.geometry_management.update(timestamp, self.motion_management[timestamp], det)
        self.score_management.update(timestamp, self.motion_management[timestamp], det)
        self.life_management.update(timestamp, self.score_management, det)
        # DE-FastPoly: update last measured v_r if valid
        if det is not None and det.get('has_valid_vr', False):
            self.last_vr = det.get('radial_vel', 0.0)
        if det is not None and ('radar_rcs' in det):
            rcs = det.get('radar_rcs', None)
            npts = int(det.get('vr_n_pts', 0)) if det.get('vr_n_pts', None) is not None else 0
            if rcs is not None and np.isfinite(float(rcs)) and npts > 0:
                self.last_rcs = float(rcs)
                self.last_rcs_npts = npts
        
    def __getitem__(self, item) -> FrameObject:
        return self.motion_management[item]
    
    def __len__(self) -> int:
        return len(self.motion_management)
    
    def __repr__(self) -> str:
        repr_str = 'tracklet status: {}, id: {}, score: {}, state: {}'
        return repr_str.format(self.life_management, self.tracking_id, 
                               self.score_management[self.timestamp],
                               self.motion_management[self.timestamp])
