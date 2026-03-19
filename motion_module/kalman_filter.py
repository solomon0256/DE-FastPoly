"""
kalman filter for trajectory state(motion state) estimation
Two implemented KF version (LKF, EKF)
Three core functions for each model: state init, state predict and state update
Linear Kalman Filter for CA, CV Model, Extend Kalman Filter for CTRA, CTRV, Bicycle Model
Ref: https://en.wikipedia.org/wiki/Kalman_filter
"""
import pdb
import numpy as np

from .nusc_object import FrameObject
from .motion_model import CA, CTRA, BICYCLE, CV, CTRV
from pre_processing import arraydet2box
from utils.math import warp_to_pi
from utils import doppler_diag
import math


class KalmanFilter:
    """kalman filter interface
    """
    def __init__(self, timestamp: int, config: dict, track_id: int, det_infos: dict) -> None:
        # init basic infos, no control input
        self.seq_id = det_infos['seq_id']
        self.initstamp = self.timestamp = timestamp
        self.tracking_id, self.class_label = track_id, det_infos['np_array'][-1]
        self.model = config['motion_model']['model'][self.class_label]
        self.dt = config['basic']['LiDAR_interval']
        self.has_velo, self.has_geofilter = config['basic']['has_velo'], config['geometry_model']['use'][self.class_label]
        # init FrameObject for each frame
        self.state, self.frame_objects = None, {}
    
    def initialize(self, det: dict) -> None:
        """initialize the filter parameters
        Args:
            det (dict): detection infos under different data format.
            {
                'nusc_box': NuscBox,
                'np_array': np.array,
                'has_velo': bool, whether the detetor has velocity info
            }
        """
        pass
    
    def predict(self, timestamp: int) -> None:
        """predict tracklet at each frame
        Args:
            timestamp (int): current frame id
        """
        pass
    
    def update(self, timestamp: int, det: dict = None) -> None:
        """update tracklet motion and geometric state
        Args:
            timestamp (int): current frame id
            det (dict, optional): same as self.init. Defaults to None.
        """
        pass
        
    def addFrameObject(self, timestamp: int, tra_info: dict, mode: str = None) -> None:
        """add predict/update tracklet state to the frameobjects, data 
        format is also implemented in this function.
        frame_objects: {
            frame_id: FrameObject
        }
        Args:
            timestamp (int): current frame id
            tra_info (dict): Trajectory state estimated by Kalman filter, 
            {
                'exter_state': np.array, for output file. 
                               [x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), tra_score, class_label]
                'inner_state': np.array, for state estimation. 
                               varies by motion model
                'cov_mat': np.mat, [2, 2], for score estimation.
            }
            mode (str, optional): stage of adding objects, 'update', 'predict'. Defaults to None.
        """
        # corner case, no tra info
        if mode is None: return
        
        # data format conversion
        inner_info, exter_info = tra_info['inner_state'], tra_info['exter_state']
        extra_info, init_geo = np.array([self.tracking_id, self.seq_id, timestamp]), False if self.has_geofilter else True
        box_info, bm_info, norm_bm_info = arraydet2box(exter_info, np.array([self.tracking_id]), init_geo)

        # update each frame infos 
        if mode == 'update':
            frame_object = self.frame_objects[timestamp]
            frame_object.update_bms, frame_object.update_norm_bms, frame_object.update_box = bm_info[0], norm_bm_info[0], box_info[0]
            frame_object.update_state, frame_object.update_infos = inner_info, np.append(exter_info, extra_info)
            frame_object.update_cov = tra_info['cov_mat']
        elif mode == 'predict':
            frame_object = FrameObject()
            frame_object.predict_bms, frame_object.predict_norm_bms, frame_object.predict_box = bm_info[0], norm_bm_info[0], box_info[0]
            frame_object.predict_state, frame_object.predict_infos = inner_info, np.append(exter_info, extra_info)
            frame_object.predict_cov = tra_info['cov_mat']
            self.frame_objects[timestamp] = frame_object
        else: raise Exception('mode must be update or predict')
    
    def getOutputInfo(self, state: np.mat) -> np.array:
        """convert state vector in the filter to the output format
        Note that, tra score will be process later
        Args:
            state (np.mat): [state dim, 1], predict or update state estimated by the filter

        Returns:
            np.array: [14(fix), 1], predict or update state under output file format
            output format: [x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), tra_score, class_label]
        """
        
        # return state vector except tra score and tra class
        inner_state = self.model.getOutputInfo(state)
        assert inner_state.shape[0] == 12, "The number of output states must satisfy 12"

        return np.append(inner_state, np.array([-1, self.class_label]))
    
    def __getitem__(self, item) -> FrameObject:
        return self.frame_objects[item]

    def __len__(self) -> int:
        return len(self.frame_objects)


class LinearKalmanFilter(KalmanFilter):
    """Linear Kalman Filter for linear motion model, such as CV and CA
    """
    def __init__(self, timestamp: int, config: dict, track_id: int, det_infos: dict) -> None:
        # init basic infos
        super(LinearKalmanFilter, self).__init__(timestamp, config, track_id, det_infos)
        # DE-FastPoly: doppler config
        doppler_cfg = config.get('doppler', {})
        self.has_doppler = doppler_cfg.get('use_doppler_obs', False)
        self.use_doppler_init = doppler_cfg.get('use_doppler_init', False)
        self.doppler_init_rho = doppler_cfg.get('doppler_init_rho', 1.0)
        self.use_range_adaptive = doppler_cfg.get('use_range_adaptive', False)
        self.range_noise_k = doppler_cfg.get('range_noise_k', 0.001)
        self.doppler_noise = doppler_cfg.get('doppler_noise', 0.5)
        # Radar-Informed Adaptive Noise (quality-aware measurement covariance)
        self.use_adaptive_noise = doppler_cfg.get('use_adaptive_noise', False)
        self.adapt_range_ref = doppler_cfg.get('adapt_range_ref', 30.0)
        self.adapt_npts_ref = doppler_cfg.get('adapt_npts_ref', 5.0)
        self.adapt_k_range = doppler_cfg.get('adapt_k_range', 0.25)
        self.adapt_k_score = doppler_cfg.get('adapt_k_score', 0.35)
        self.adapt_k_vr = doppler_cfg.get('adapt_k_vr', 0.20)
        self.adapt_k_npts = doppler_cfg.get('adapt_k_npts', 0.15)
        self.adapt_min_scale = doppler_cfg.get('adapt_min_scale', 0.5)
        self.adapt_max_scale = doppler_cfg.get('adapt_max_scale', 2.5)
        self.adapt_pos_only = doppler_cfg.get('adapt_pos_only', False)  # only scale position R dims
        # DE-FastPoly: A-min pseudo-observation config
        self.use_doppler_pseudo = doppler_cfg.get('use_doppler_pseudo', False)
        self.pseudo_sigma = doppler_cfg.get('pseudo_sigma', 20.0)
        self.pseudo_vr_thre = doppler_cfg.get('pseudo_vr_thre', 2.0)
        self.pseudo_allowed_cls = set(doppler_cfg.get('pseudo_allowed_cls', [1, 2]))
        # set motion model, default Constant Acceleration(CA) for LKF
        if self.model in ['CV', 'CA']:
            model_cls = globals()[self.model]
            if self.model == 'CV':
                self.model = model_cls(self.has_velo, self.has_geofilter, self.dt, has_doppler=self.has_doppler, use_doppler_init=self.use_doppler_init, doppler_init_rho=self.doppler_init_rho)
            else:
                self.model = model_cls(self.has_velo, self.has_geofilter, self.dt)
        else:
            self.model = globals()['CA'](self.has_velo, self.has_geofilter, self.dt)
        # Transition and Observation Matrices are fixed in the LKF
        self.initialize(det_infos)
        
    def initialize(self, det_infos: dict) -> None:
        # state transition
        self.F = self.model.getTransitionF()
        self.Q = self.model.getProcessNoiseQ(self.class_label)
        self.SD = self.model.getStateDim()
        self.P = self.model.getInitCovP(self.class_label)

        # state to measurement transition
        self.R_base = self.model.getMeaNoiseR(self.class_label, self.doppler_noise)
        self.R = self.R_base.copy()
        self.H = self.model.getMeaStateH()

        # get different data format tracklet's state
        self.state = self.model.getInitState(det_infos)
        tra_infos = {
            'inner_state': self.state,
            'exter_state': det_infos['np_array'],
            'cov_mat': self.P[:2, :2],
        }
        self.addFrameObject(self.timestamp, tra_infos, 'predict')
        self.addFrameObject(self.timestamp, tra_infos, 'update')
    
    def predict(self, timestamp: int) -> None:
        # predict state and errorcov
        self.state = self.F * self.state
        self.P = self.F * self.P * self.F.T + self.Q

        # convert the state in filter to the output format
        self.model.warpStateYawToPi(self.state)
        output_info = self.getOutputInfo(self.state)
        tra_infos = {
            'inner_state': self.state,
            'exter_state': output_info,
            'cov_mat': self.P[:2, :2],
        }
        self.addFrameObject(timestamp, tra_infos, 'predict')
        
    def update(self, timestamp: int, det: dict = None) -> None:
        # corner case, no det for updating
        if det is None: return

        # Check if this detection has valid radial velocity
        has_valid_vr = det.get('has_valid_vr', False) if det else False
        if self.has_doppler:
            doppler_diag.record_a_update(int(self.class_label), bool(has_valid_vr))

        # DE-FastPoly: recompute H dynamically for doppler observation (Contribution A)
        if self.has_doppler:
            self.H = self.model.getMeaStateH(self.state)
            if not has_valid_vr:
                # No valid v_r: drop the doppler row from H
                self.H = self.H[:-1, :]

        # DE-FastPoly: range-adaptive measurement noise (Contribution B)
        # Additive scaling on position dims only: R_pos += k * rÂ²
        if self.use_range_adaptive:
            x, y = self.state[0, 0], self.state[1, 0]
            r_sq = x**2 + y**2
            self.R = self.R_base.copy()
            self.R[0, 0] += self.range_noise_k * r_sq
            self.R[1, 1] += self.range_noise_k * r_sq
        else:
            self.R = self.R_base

        # Drop doppler dimension from R when v_r is invalid
        if self.has_doppler and not has_valid_vr:
            self.R = self.R[:-1, :-1]

        # Radar-Informed Adaptive Noise:
        # Good observation quality (high score, valid v_r, many radar points) -> lower R.
        # Hard condition (far range) -> higher R.
        if self.use_adaptive_noise:
            score = 0.5
            if det is not None:
                if 'np_array' in det and len(det['np_array']) > 12:
                    score = float(det['np_array'][12])
                elif 'nusc_box' in det and hasattr(det['nusc_box'], 'score'):
                    score = float(det['nusc_box'].score)
            score = float(np.clip(score, 0.0, 1.0))

            if det is not None and 'np_array' in det and len(det['np_array']) >= 2:
                dx, dy = float(det['np_array'][0]), float(det['np_array'][1])
            else:
                dx, dy = float(self.state[0, 0]), float(self.state[1, 0])
            r = np.hypot(dx, dy)
            range_penalty = (r / max(self.adapt_range_ref, 1e-6)) ** 2

            vr_quality = 1.0 if has_valid_vr else 0.0
            n_pts = det.get('vr_n_pts', None) if det is not None else None
            if n_pts is None:
                npts_quality = 0.0
            else:
                npts_quality = 1.0 - np.exp(-float(max(n_pts, 0.0)) / max(self.adapt_npts_ref, 1e-6))

            scale = (
                1.0
                + self.adapt_k_range * range_penalty
                - self.adapt_k_score * score
                - self.adapt_k_vr * vr_quality
                - self.adapt_k_npts * npts_quality
            )
            scale = float(np.clip(scale, self.adapt_min_scale, self.adapt_max_scale))
            # Position-only scaling: only scale position dims (first 2 rows/cols),
            # leave velocity and yaw noise untouched.
            # This prevents AN from disrupting well-calibrated velocity noise
            # when the detector already provides velocity (e.g., CenterPoint).
            if self.adapt_pos_only:
                self.R[0, 0] *= scale
                self.R[1, 1] *= scale
            else:
                self.R = self.R * scale

        # update state and errorcov
        meas_info = self.model.getMeasureInfo(det)
        if self.has_doppler and not has_valid_vr:
            meas_info = meas_info[:-1, :]  # drop v_r from measurement
        _res = meas_info - self.H * self.state
        # Warp yaw residual to [-pi, pi). When doppler row is dropped,
        # yaw is last element; when present, yaw is second-to-last.
        if self.has_doppler and has_valid_vr:
            _res[-2, 0] = warp_to_pi(_res[-2, 0])
        else:
            _res[-1, 0] = warp_to_pi(_res[-1, 0])
        _S = self.H * self.P * self.H.T + self.R
        if self.has_doppler and has_valid_vr:
            doppler_diag.record_a_innovation(
                int(self.class_label),
                float(_res[-1, 0]),
                float(_S[-1, -1]),
            )
        _KF_GAIN = self.P * self.H.T * _S.I

        self.state += _KF_GAIN * _res
        self.P = (np.mat(np.identity(self.SD)) - _KF_GAIN * self.H) * self.P

        # DE-FastPoly A-min: 1D radial pseudo-observation update
        # Done AFTER the normal KF update, as a separate weak constraint
        if self.use_doppler_pseudo and det is not None:
            _has_vr = det.get('has_valid_vr', False)
            _vr = det.get('radial_vel', 0.0)
            _cls = int(self.class_label)
            if _has_vr and abs(_vr) > self.pseudo_vr_thre and _cls in self.pseudo_allowed_cls:
                det_arr = det.get('np_array', None)
                if det_arr is not None:
                    _xd, _yd = float(det_arr[0]), float(det_arr[1])
                    _rd = np.sqrt(_xd**2 + _yd**2)
                    if _rd > 1e-6:
                        _rx, _ry = _xd / _rd, _yd / _rd
                        _Hd = np.mat(np.zeros((1, self.SD)))
                        if self.model.has_geofilter:
                            _Hd[0, 2], _Hd[0, 3] = _rx, _ry
                        else:
                            _Hd[0, 6], _Hd[0, 7] = _rx, _ry
                        _Rd = np.mat([[self.pseudo_sigma ** 2]])
                        _zd = np.mat([[_vr]])
                        _resd = _zd - _Hd * self.state
                        _Sd = _Hd * self.P * _Hd.T + _Rd
                        _Kd = self.P * _Hd.T * _Sd.I
                        self.state += _Kd * _resd
                        self.P = (np.mat(np.eye(self.SD)) - _Kd * _Hd) * self.P

        # output updated state to the result file
        self.model.warpStateYawToPi(self.state)
        output_info = self.getOutputInfo(self.state)
        tra_infos = {
            'inner_state': self.state,
            'exter_state': output_info,
            'cov_mat': self.P[:2, :2],
        }
        self.addFrameObject(timestamp, tra_infos, 'update')


class ExtendKalmanFilter(KalmanFilter):
    def __init__(self, timestamp: int, config: dict, track_id: int, det_infos: dict) -> None:
        super().__init__(timestamp, config, track_id, det_infos)
        doppler_cfg = config.get('doppler', {})
        self.has_doppler = doppler_cfg.get('use_doppler_obs', False)
        # Radar-Informed Adaptive Noise for EKF classes
        self.use_adaptive_noise = doppler_cfg.get('use_adaptive_noise', False)
        self.adapt_range_ref = doppler_cfg.get('adapt_range_ref', 30.0)
        self.adapt_npts_ref = doppler_cfg.get('adapt_npts_ref', 5.0)
        self.adapt_k_range = doppler_cfg.get('adapt_k_range', 0.25)
        self.adapt_k_score = doppler_cfg.get('adapt_k_score', 0.35)
        self.adapt_k_vr = doppler_cfg.get('adapt_k_vr', 0.20)
        self.adapt_k_npts = doppler_cfg.get('adapt_k_npts', 0.15)
        self.adapt_min_scale = doppler_cfg.get('adapt_min_scale', 0.5)
        self.adapt_max_scale = doppler_cfg.get('adapt_max_scale', 2.5)
        self.adapt_pos_only = doppler_cfg.get('adapt_pos_only', False)
        # set motion model, default Constant Acceleration and Turn Rate(CTRA) for EKF
        self.model = globals()[self.model](self.has_velo, self.has_geofilter, self.dt) if self.model in ['CTRV', 'CTRA', 'BICYCLE'] \
                     else globals()['CTRA'](self.has_velo, self.has_geofilter, self.dt)
        # Transition and Observation Matrices are changing in the EKF
        self.initialize(det_infos)
    
    def initialize(self, det_infos: dict) -> None:
        # init errorcov categoty-specific
        self.SD, self.MD = self.model.getStateDim(), self.model.getMeasureDim()
        self.Identity_MD, self.Identity_SD = np.mat(np.identity(self.MD)), np.mat(np.identity(self.SD))
        self.P = self.model.getInitCovP(self.class_label)
        
        # set noise matrix(fixed)
        self.Q = self.model.getProcessNoiseQ(self.class_label)
        self.R = self.model.getMeaNoiseR(self.class_label)

        # get different data format tracklet's state
        self.state = self.model.getInitState(det_infos)
        tra_infos = {
            'inner_state': self.state,
            'exter_state': det_infos['np_array'],
            'cov_mat': self.P[:2, :2],
        }
        self.addFrameObject(self.timestamp, tra_infos, 'predict')
        self.addFrameObject(self.timestamp, tra_infos, 'update')
        
    def predict(self, timestamp: int) -> None:
        # get jacobian matrix F using the final estimated state of the previous frame
        self.F = self.model.getTransitionF(self.state)
        
        # state and errorcov transition
        self.state = self.model.stateTransition(self.state)

        self.P = self.F * self.P * self.F.T + self.Q
        # convert the state in filter to the output format
        self.model.warpStateYawToPi(self.state)
        output_info = self.getOutputInfo(self.state)
        tra_infos = {
            'inner_state': self.state,
            'exter_state': output_info,
            'cov_mat': self.P[:2, :2],
        }
        self.addFrameObject(timestamp, tra_infos, 'predict')
    
    def update(self, timestamp: int, det: dict = None) -> None:
        # corner case, no det for updating
        if det is None: return
        if self.has_doppler:
            doppler_diag.record_a_ekf_path(int(self.class_label))
        
        # get measure infos for updating, and project state into meausre space
        meas_info = self.model.getMeasureInfo(det)
        state_info = self.model.StateToMeasure(self.state)
        
        # get state residual, and warp angle diff inplace
        _res = meas_info - state_info
        self.model.warpResYawToPi(_res)
        
        # get jacobian matrix H using the predict state
        self.H = self.model.getMeaStateH(self.state)

        # Radar-Informed Adaptive Noise on EKF R (same rule as LKF).
        R_use = self.R
        if self.use_adaptive_noise and det is not None:
            score = 0.5
            if 'np_array' in det and len(det['np_array']) > 12:
                score = float(det['np_array'][12])
            elif 'nusc_box' in det and hasattr(det['nusc_box'], 'score'):
                score = float(det['nusc_box'].score)
            score = float(np.clip(score, 0.0, 1.0))

            if 'np_array' in det and len(det['np_array']) >= 2:
                dx, dy = float(det['np_array'][0]), float(det['np_array'][1])
            else:
                dx, dy = float(self.state[0, 0]), float(self.state[1, 0])
            r = np.hypot(dx, dy)
            range_penalty = (r / max(self.adapt_range_ref, 1e-6)) ** 2

            has_valid_vr = bool(det.get('has_valid_vr', False))
            vr_quality = 1.0 if has_valid_vr else 0.0
            n_pts = det.get('vr_n_pts', None)
            if n_pts is None:
                npts_quality = 0.0
            else:
                npts_quality = 1.0 - np.exp(-float(max(n_pts, 0.0)) / max(self.adapt_npts_ref, 1e-6))

            scale = (
                1.0
                + self.adapt_k_range * range_penalty
                - self.adapt_k_score * score
                - self.adapt_k_vr * vr_quality
                - self.adapt_k_npts * npts_quality
            )
            scale = float(np.clip(scale, self.adapt_min_scale, self.adapt_max_scale))
            if self.adapt_pos_only:
                R_use = self.R.copy()
                R_use[0, 0] *= scale
                R_use[1, 1] *= scale
            else:
                R_use = self.R * scale

        # obtain KF gain and update state and errorcov
        _S = self.H * self.P * self.H.T + R_use
        # try:
        #     # obtain the cholesky factor U (upper matrix)
        #     _U = scipy.linalg.cholesky(a=_S, 
        #                                check_finite=False)     
            
        #     # speed up the inverse process (a * x = b -> U * U-1 = I)
        #     _U_INV = np.mat(scipy.linalg.solve_triangular(a=_U, 
        #                                                   b=self.Identity_MD, 
        #                                                   check_finite=False))
                    
        #     _KF_GAIN = self.P * self.H.T * _U_INV * _U_INV.T
        # except:
        #     pdb.set_trace()
        _KF_GAIN = self.P * self.H.T * _S.I
        _I_KH = self.Identity_SD - _KF_GAIN * self.H
        
        self.state += _KF_GAIN * _res
        # self.P = _I_KH * self.P * _I_KH.T + _KF_GAIN * self.R * _KF_GAIN.T
        self.P = _I_KH * self.P
        
        # output updated state to the result file
        self.model.warpStateYawToPi(self.state)
        output_info = self.getOutputInfo(self.state)
        tra_infos = {
            'inner_state': self.state,
            'exter_state': output_info,
            'cov_mat': self.P[:2, :2],
        }
        self.addFrameObject(timestamp, tra_infos, 'update')
        
        
        
        
        
        
        
            
        
        
        
        
        
        
        
    
    
        
