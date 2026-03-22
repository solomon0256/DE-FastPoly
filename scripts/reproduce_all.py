"""
reproduce_all.py - Reproduce ALL paper results in one command

Runs every experiment used in the paper and verifies against expected values.
If any result deviates beyond tolerance, prints a WARNING.

Usage:
  python reproduce_all.py

Expected runtime: ~3 minutes
"""
import sys, os, copy, time
import numpy as np
from collections import defaultdict

FASTPOLY_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FASTPOLY_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
from tracking.nusc_tracker import Tracker
from pre_processing.nusc_data_conversion import arraydet2box
from run_vod import (read_radar_pointcloud, get_doppler_for_box,
                     build_frame_info, load_frame_ids, parse_kitti_label,
                     make_np_det, LABEL_DIR)
from run_vod_det import parse_det_label, make_np_det_scored, DET_DIR
from eval_vod import load_gt, evaluate, find_moving_track_ids

# ============================================================
# FROZEN PARAMETERS — DO NOT CHANGE
# These are the final paper parameters. Any change invalidates results.
# ============================================================
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'vod_config.yaml')
ALPHA = 0.95
BETA = 0.05
SIGMA = 10.0
VR_THRE = {0: 1.0, 1: 0.3, 2: 0.5}
VR_ZERO = {0: 0.0, 1: 0.0, 2: 0.0}
DET_SCORE_THRE = 0.2

CLASS_NAMES = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}

# Expected results (for verification)
EXPECTED = {
    'oracle': {
        'Baseline':    {'mota': 0.854, 'ids': 158},
        'C only':      {'mota': 0.867, 'ids': 165},
        'D only':      {'mota': 0.880, 'ids': 179},
        'C+D':         {'mota': 0.887, 'ids': 188},
        'D nofilter':  {'mota': 0.893, 'ids': 164},
        'CD nofilter': {'mota': 0.893, 'ids': 165},
    },
    'detector': {
        'Baseline': {'mota': 0.282, 'ids': 148},
        'C only':   {'mota': 0.278, 'ids': 152},
        'D only':   {'mota': 0.294, 'ids': 152},
        'C+D':      {'mota': 0.299, 'ids': 154},
    },
}
MOTA_TOL = 0.002  # tolerance for MOTA comparison
IDS_TOL = 3       # tolerance for IDS comparison


def make_doppler_cfg(C=False, D=False, vr_class_thre=None):
    cfg = {
        'use_doppler_obs': False,
        'use_doppler_assoc': D,
        'use_doppler_gate': False,
        'use_doppler_init': C,
        'use_range_adaptive': False,
        'use_doppler_discount': False,
        'assoc_alpha': ALPHA,
        'assoc_beta': BETA,
        'doppler_sigma': SIGMA,
        'doppler_gate_thre': 3.0,
        'range_noise_k': 0.00003,
        'doppler_noise': 10.0,
        'doppler_discount': 0.7,
        'doppler_discount_thre': 0.7,
    }
    if vr_class_thre is not None:
        cfg['vr_class_thre'] = vr_class_thre
    else:
        cfg['vr_class_thre'] = VR_THRE.copy()
    return cfg


def preload_data(sequences, mode='oracle'):
    all_fids = [fid for seq in sequences for fid in seq]
    preloaded = {}
    for fid in all_fids:
        if mode == 'oracle':
            label_path = os.path.join(LABEL_DIR, f'{fid:05d}.txt')
            dets = []
            if os.path.exists(label_path):
                with open(label_path) as lf:
                    for line in lf:
                        det = parse_kitti_label(line)
                        if det is not None:
                            dets.append(det)
        else:
            det_path = os.path.join(DET_DIR, f'{fid:05d}.txt')
            dets = []
            if os.path.exists(det_path):
                with open(det_path) as lf:
                    for line in lf:
                        det = parse_det_label(line, DET_SCORE_THRE)
                        if det is not None:
                            dets.append(det)

        pc = read_radar_pointcloud(fid)
        for det in dets:
            vr, found, _ = get_doppler_for_box(pc, det['x'], det['y'], det['z'],
                                            det['w'], det['l'], det['h'], det['ry'])
            det['radial_vel'] = vr
            det['has_valid_vr'] = found

        entry = {'dets': dets}
        if dets:
            if mode == 'oracle':
                np_dets = np.array([make_np_det(d) for d in dets])
            else:
                np_dets = np.array([make_np_det_scored(d) for d in dets])
            box_dets, bm_dets, norm_bm_dets = arraydet2box(np_dets, init_geo=True)
            entry['np_dets'] = np_dets
            entry['box_dets'] = box_dets
            entry['bm_dets'] = bm_dets
            entry['norm_bm_dets'] = norm_bm_dets
            entry['radial_vels'] = np.array([d['radial_vel'] for d in dets])
            entry['valid_vr_mask'] = np.array([d['has_valid_vr'] for d in dets])
        preloaded[fid] = entry
    return preloaded


def run_tracker(cfg, sequences, preloaded):
    pred_by_frame = defaultdict(list)
    for seq_idx, seq_frames in enumerate(sequences):
        tracker = Tracker(cfg)
        tracker.reset()
        for i, fid in enumerate(seq_frames):
            is_first = (i == 0)
            dets = preloaded[fid]['dets']
            if not dets:
                data_info = build_frame_info(fid, dets, seq_id=seq_idx, is_first=is_first)
            else:
                data_info = {
                    'frame_id': fid, 'seq_id': seq_idx,
                    'is_first_frame': is_first, 'no_dets': False,
                    'det_num': len(dets), 'np_dets': preloaded[fid]['np_dets'],
                    'box_dets': preloaded[fid]['box_dets'],
                    'np_dets_bottom_corners': preloaded[fid]['bm_dets'],
                    'np_dets_norm_corners': preloaded[fid]['norm_bm_dets'],
                    'has_velo': False,
                    'radial_vels': preloaded[fid]['radial_vels'],
                    'valid_vr_mask': preloaded[fid]['valid_vr_mask'],
                }
            tracker.tracking(data_info)
            if data_info.get('no_val_track_result') or 'np_track_res' not in data_info:
                continue
            np_track_res = np.array(data_info['np_track_res'])
            for row in np_track_res:
                pred_by_frame[fid].append({
                    'track_id': int(row[14]),
                    'class_id': int(row[13]),
                    'x': row[0], 'y': row[1], 'z': row[2],
                })
    return pred_by_frame


def verify(name, mota, ids, expected, mode):
    exp = expected.get(mode, {}).get(name, None)
    if exp is None:
        return "  (no expected value)"
    warnings = []
    if abs(mota - exp['mota']) > MOTA_TOL:
        warnings.append(f"MOTA {mota:.3f} != expected {exp['mota']:.3f}")
    if abs(ids - exp['ids']) > IDS_TOL:
        warnings.append(f"IDS {ids} != expected {exp['ids']}")
    if warnings:
        return f"  WARNING: {', '.join(warnings)}"
    return "  OK"


def main():
    t_start = time.time()
    base_cfg = yaml.safe_load(open(CONFIG_PATH, encoding='utf-8'))
    gt_by_frame, frame_ids = load_gt('val')
    sequences = load_frame_ids('val')
    print(f"VoD val: {len(frame_ids)} frames, {len(sequences)} sequences")
    print(f"Parameters: alpha={ALPHA}, beta={BETA}, sigma={SIGMA}")
    print(f"Det score_thre: {DET_SCORE_THRE}")

    all_pass = True

    # ===== ORACLE =====
    print(f"\n{'='*70}")
    print(f"  ORACLE (GT boxes)")
    print(f"{'='*70}")
    preloaded_gt = preload_data(sequences, mode='oracle')

    oracle_configs = [
        ('Baseline',    dict(C=False, D=False)),
        ('C only',      dict(C=True,  D=False)),
        ('D only',      dict(C=False, D=True)),
        ('C+D',         dict(C=True,  D=True)),
        ('D nofilter',  dict(C=False, D=True,  vr_class_thre=VR_ZERO)),
        ('CD nofilter', dict(C=True,  D=True,  vr_class_thre=VR_ZERO)),
    ]

    print(f"  {'Config':<16} {'MOTA':>7} {'IDS':>5} {'TP':>6} {'FP':>6} {'FN':>6}  Status")
    print(f"  {'-'*65}")
    for name, kwargs in oracle_configs:
        cfg = copy.deepcopy(base_cfg)
        cfg['doppler'] = make_doppler_cfg(**kwargs)
        pred = run_tracker(cfg, sequences, preloaded_gt)
        metrics = evaluate(gt_by_frame, pred, frame_ids)
        m = metrics[None]
        mota = 1.0 - (m['fp'] + m['fn'] + m['ids']) / max(m['n_gt'], 1)
        ids = m['ids']
        status = verify(name, mota, ids, EXPECTED, 'oracle')
        if 'WARNING' in status:
            all_pass = False
        print(f"  {name:<16} {mota:>7.3f} {ids:>5d} {m['tp']:>6d} {m['fp']:>6d} {m['fn']:>6d}{status}")

    # ===== DETECTOR =====
    print(f"\n{'='*70}")
    print(f"  DETECTOR (PointPillars, score_thre={DET_SCORE_THRE})")
    print(f"{'='*70}")
    preloaded_det = preload_data(sequences, mode='detector')

    det_configs = [
        ('Baseline', dict(C=False, D=False)),
        ('C only',   dict(C=True,  D=False)),
        ('D only',   dict(C=False, D=True)),
        ('C+D',      dict(C=True,  D=True)),
    ]

    print(f"  {'Config':<16} {'MOTA':>7} {'IDS':>5} {'TP':>6} {'FP':>6} {'FN':>6}  Status")
    print(f"  {'-'*65}")
    for name, kwargs in det_configs:
        cfg = copy.deepcopy(base_cfg)
        cfg['doppler'] = make_doppler_cfg(**kwargs)
        pred = run_tracker(cfg, sequences, preloaded_det)
        metrics = evaluate(gt_by_frame, pred, frame_ids)
        m = metrics[None]
        mota = 1.0 - (m['fp'] + m['fn'] + m['ids']) / max(m['n_gt'], 1)
        ids = m['ids']
        status = verify(name, mota, ids, EXPECTED, 'detector')
        if 'WARNING' in status:
            all_pass = False
        print(f"  {name:<16} {mota:>7.3f} {ids:>5d} {m['tp']:>6d} {m['fp']:>6d} {m['fn']:>6d}{status}")

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    if all_pass:
        print(f"  ALL RESULTS MATCH EXPECTED VALUES ({elapsed:.0f}s)")
    else:
        print(f"  SOME RESULTS DO NOT MATCH — CHECK WARNINGS ABOVE ({elapsed:.0f}s)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
