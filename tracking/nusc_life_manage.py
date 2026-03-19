"""
Count-based Trajectory Lifecycle Management Module.
Trajectory state(tentative/active/death) transition and tracking score punish
"""
import pdb

from .nusc_score_manage import ScoreManagement

class LifeManagement:
    def __init__(self, timestamp: int, config: dict, class_label: int):
        self.cfg = config['life_cycle']
        self.curr_time = self.init_time = timestamp
        self.time_since_update, self.hit, self.state_jump = 0, 1, False
        self.delete_thre = self.cfg['score']['delete_thre'][class_label]
        self.min_hit, self.max_age = self.cfg['basic']['min_hit'][class_label], self.cfg['basic']['max_age'][class_label]
        self.state = 'active' if self.min_hit <= 1 or timestamp <= self.min_hit else 'tentative'
        self.termination = config['life_cycle']['score']['termination']
        # DE-FastPoly: radar-informed lifecycle
        doppler_cfg = config.get('doppler', {})
        self.use_radar_lifecycle = doppler_cfg.get('use_radar_lifecycle', False)
        self.radar_confirm_npts = doppler_cfg.get('radar_confirm_npts', 3)
        self.radar_fast_delete_frames = doppler_cfg.get('radar_fast_delete_frames', 3)
        self.radar_bonus_age = doppler_cfg.get('radar_bonus_age', 2)
        # Track cumulative radar point count
        self.total_radar_pts = 0
        self.frames_no_radar = 0

    def predict(self, timestamp: int) -> None:
        """
        predict tracklet lifecycle
        :param timestamp: int, current timestamp, frame id
        """
        self.curr_time = timestamp
        self.time_since_update += 1

    def update(self, timestamp: int, score_mgt: ScoreManagement, det = None) -> None:
        """
        update tracklet lifecycle status, switch tracklet's state (tentative/dead/active)
        :param timestamp: int, current timestamp, frame id
        :param score_mgt: ScoreManagement, score management of current tracklet
        :param det: matched detection at current frame
        """
        if det is not None:
            self.hit += 1
            self.time_since_update = 0

        # DE-FastPoly: track radar point statistics
        if self.use_radar_lifecycle and det is not None:
            n_pts = det.get('vr_n_pts', 0)
            self.total_radar_pts += n_pts
            if n_pts == 0:
                self.frames_no_radar += 1
            else:
                self.frames_no_radar = 0

        if self.state == 'tentative':
            # DE-FastPoly: fast confirmation if detection has many radar points
            if self.use_radar_lifecycle and det is not None:
                n_pts = det.get('vr_n_pts', 0)
                if n_pts >= self.radar_confirm_npts and self.hit >= 2:
                    self.state, self.state_jump = 'active', True
                    return

            if (self.hit >= self.min_hit) or (timestamp <= self.min_hit):
                self.state, self.state_jump = 'active', True
            elif self.time_since_update > 0:
                self.state = 'dead'
            else: self.state_jump = False
        elif self.state == 'active':

            if self.termination == 'average':
                score_kill = (score_mgt.trk_avg_score < self.delete_thre)
            elif self.termination == 'latest':
                score_kill = (score_mgt[timestamp].final_score < self.delete_thre)
            else: raise Exception("termination only has two functions")

            # DE-FastPoly: radar-informed max_age bonus
            # Tracks with strong radar evidence survive longer
            effective_max_age = self.max_age
            if self.use_radar_lifecycle and self.total_radar_pts > 0:
                effective_max_age = self.max_age + self.radar_bonus_age

            # DE-FastPoly: fast delete if no radar points for consecutive frames
            radar_kill = False
            if self.use_radar_lifecycle and self.frames_no_radar >= self.radar_fast_delete_frames:
                if self.time_since_update >= 2:
                    radar_kill = True

            if (self.time_since_update >= effective_max_age) or score_kill or radar_kill:
                self.state = 'dead'
            else: self.state_jump = False
        else: raise Exception("dead trajectory cannot be updated")
        
    def __repr__(self) -> str:
        repr_str = 'init_timestamp: {}, time_since_update: {}, hit: {}, state: {}'
        return repr_str.format(self.init_time, self.time_since_update, self.hit, self.state)




