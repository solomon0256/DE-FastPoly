"""
paper_ablation.py - Final ablation for ISIE 2026 paper

Runs 6 core ablation configs + 4 threshold sensitivity configs on VoD val.
All configs use GT boxes (oracle detection / tracker-only evaluation).
All configs share identical eval pipeline (eval_vod.py).

Usage:
  python paper_ablation.py                    # run all
  python paper_ablation.py --only-ablation    # only 6-row ablation
  python paper_ablation.py --only-sensitivity # only threshold sensitivity
"""
import sys, os, copy, time, argparse
import numpy as np
from collections import defaultdict

FASTPOLY_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FASTPOLY_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
from tracking.nusc_tracker import Tracker
from pre_processing.nusc_data_conversion import arraydet2box
from run_vod import (read_radar_pointcloud, get_doppler_for_box,
                     get_radar_stats_for_box,
                     build_frame_info, load_frame_ids, parse_kitti_label,
                     make_np_det, LABEL_DIR)
from eval_vod import load_gt, evaluate, find_moving_track_ids

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'vod_config.yaml')
CLASS_NAMES = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}

# Final parameters (frozen, do not change)
FINAL_ALPHA = 0.95
FINAL_BETA = 0.05
FINAL_SIGMA = 10.0
FINAL_VR_THRE = {0: 1.0, 1: 0.3, 2: 0.5}  # Car, Ped, Cyc
ZERO_VR_THRE = {0: 0.0, 1: 0.0, 2: 0.0}   # No filter


def make_doppler_cfg(C=False, D=False, AN=False, RCS=False, vr_class_thre=None):
    """Build doppler config. C/D/AN/RCS toggles (A/B/E/F disabled)."""
    cfg = {
        'use_doppler_obs': False,
        'use_doppler_assoc': D,
        'use_doppler_gate': False,
        'use_doppler_init': C,
        'use_range_adaptive': False,
        'use_doppler_discount': False,
        'assoc_alpha': FINAL_ALPHA,
        'assoc_beta': FINAL_BETA,
        'doppler_sigma': FINAL_SIGMA,
        'doppler_gate_thre': 3.0,
        'range_noise_k': 0.00003,
        'doppler_noise': 10.0,
        'doppler_discount': 0.7,
        'doppler_discount_thre': 0.7,
        # AN: Adaptive Measurement Noise
        'use_adaptive_noise': AN,
        'adapt_pos_only': True,
        'adapt_k_range': 0.0,
        'adapt_k_score': 0.7,
        'adapt_k_vr': 0.0,
        'adapt_k_npts': 0.0,
        'adapt_min_scale': 0.3,
        'adapt_max_scale': 2.5,
        # RCS: RCS-Aware Association
        'use_rcs_assoc': RCS,
        'rcs_gamma': 0.005,
        'rcs_sigma': 6.0,
        'rcs_min_npts': 1,
        'rcs_score_gate': 0.3,
        'rcs_allowed_cls': [0, 1, 2],  # all classes for oracle
    }
    if vr_class_thre is not None:
        cfg['vr_class_thre'] = vr_class_thre
    else:
        cfg['vr_class_thre'] = FINAL_VR_THRE.copy()
    return cfg


def preload_gt_data(sequences):
    """Preload GT labels + radar v_r for all frames (once)."""
    all_fids = [fid for seq in sequences for fid in seq]
    print(f"Preloading {len(all_fids)} frames...")
    t0 = time.time()

    preloaded = {}
    total_dets = 0
    total_vr = 0

    for fid in all_fids:
        label_path = os.path.join(LABEL_DIR, f'{fid:05d}.txt')
        dets = []
        if os.path.exists(label_path):
            with open(label_path) as lf:
                for line in lf:
                    det = parse_kitti_label(line)
                    if det is not None:
                        dets.append(det)

        pc = read_radar_pointcloud(fid)
        for det in dets:
            vr, found, n_pts, rcs = get_radar_stats_for_box(
                pc, det['x'], det['y'], det['z'],
                det['w'], det['l'], det['h'], det['ry'])
            det['radial_vel'] = vr
            det['has_valid_vr'] = found
            det['radar_rcs'] = rcs
            det['vr_n_pts'] = n_pts
            if found:
                total_vr += 1
        total_dets += len(dets)

        entry = {'dets': dets}
        if dets:
            np_dets = np.array([make_np_det(d) for d in dets])
            box_dets, bm_dets, norm_bm_dets = arraydet2box(np_dets, init_geo=True)
            entry['np_dets'] = np_dets
            entry['box_dets'] = box_dets
            entry['bm_dets'] = bm_dets
            entry['norm_bm_dets'] = norm_bm_dets
            entry['radial_vels'] = np.array([d['radial_vel'] for d in dets])
            entry['valid_vr_mask'] = np.array([d['has_valid_vr'] for d in dets])
            entry['radar_rcs'] = np.array([d.get('radar_rcs', 0.0) for d in dets], dtype=float)
            entry['vr_n_pts'] = np.array([d.get('vr_n_pts', 0) for d in dets], dtype=float)
        preloaded[fid] = entry

    elapsed = time.time() - t0
    print(f"Preloaded in {elapsed:.1f}s: {total_dets} GT boxes, "
          f"{total_vr} with v_r ({100*total_vr/max(total_dets,1):.1f}%)")
    return preloaded


def run_tracker(cfg, sequences, preloaded):
    """Run tracker, return pred_by_frame."""
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
                    'radar_rcs': preloaded[fid].get('radar_rcs', np.array([])),
                    'vr_n_pts': preloaded[fid].get('vr_n_pts', np.array([])),
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


def format_metrics(metrics):
    """Extract formatted metrics dict from eval output."""
    result = {}
    for cls_id in [None, 0, 1, 2]:
        m = metrics[cls_id]
        name = 'OVERALL' if cls_id is None else CLASS_NAMES[cls_id]
        if m['n_gt'] == 0:
            result[name] = {}
            continue
        mota = 1.0 - (m['fp'] + m['fn'] + m['ids']) / max(m['n_gt'], 1)
        motp = m['total_dist'] / max(m['n_matches'], 1)
        result[name] = {
            'mota': mota, 'motp': motp, 'ids': m['ids'],
            'tp': m['tp'], 'fp': m['fp'], 'fn': m['fn'],
        }
    return result


def print_table(all_results, baseline_key='Baseline'):
    """Print formatted ablation table."""
    bl = all_results.get(baseline_key, {}).get('OVERALL', {})

    # Header
    print(f"\n{'='*100}")
    print(f"  {'Config':<20s} {'MOTA':>7} {'MOTP':>7} {'IDS':>5} "
          f"{'TP':>6} {'FP':>6} {'FN':>6}  "
          f"{'Car':>7} {'Ped':>7} {'Cyc':>7}  {'dMOTA':>7} {'dIDS':>5}")
    print(f"  {'-'*95}")

    for name, fm in all_results.items():
        o = fm.get('OVERALL', {})
        if not o:
            continue
        car = fm.get('Car', {}).get('mota', float('nan'))
        ped = fm.get('Pedestrian', {}).get('mota', float('nan'))
        cyc = fm.get('Cyclist', {}).get('mota', float('nan'))

        dmota = o['mota'] - bl.get('mota', 0) if bl else 0
        dids = o['ids'] - bl.get('ids', 0) if bl else 0

        print(f"  {name:<20s} {o['mota']:>7.3f} {o['motp']:>7.3f} {o['ids']:>5.0f} "
              f"{o['tp']:>6d} {o['fp']:>6d} {o['fn']:>6d}  "
              f"{car:>7.3f} {ped:>7.3f} {cyc:>7.3f}  "
              f"{dmota:>+7.3f} {dids:>+5.0f}")

    print(f"{'='*100}")

    # Per-class IDS breakdown
    print(f"\n  Per-class IDS:")
    print(f"  {'Config':<20s} {'Car':>6} {'Ped':>6} {'Cyc':>6} {'Total':>6}")
    print(f"  {'-'*50}")
    for name, fm in all_results.items():
        car_ids = fm.get('Car', {}).get('ids', 0)
        ped_ids = fm.get('Pedestrian', {}).get('ids', 0)
        cyc_ids = fm.get('Cyclist', {}).get('ids', 0)
        total_ids = fm.get('OVERALL', {}).get('ids', 0)
        print(f"  {name:<20s} {car_ids:>6.0f} {ped_ids:>6.0f} {cyc_ids:>6.0f} {total_ids:>6.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only-ablation', action='store_true')
    parser.add_argument('--only-sensitivity', action='store_true')
    parser.add_argument('--split', default='val')
    args = parser.parse_args()

    base_cfg = yaml.safe_load(open(CONFIG_PATH, encoding='utf-8'))

    # Load GT and sequences
    gt_by_frame, frame_ids = load_gt(args.split)
    moving_ids = find_moving_track_ids(gt_by_frame, frame_ids, move_thre=1.0)
    sequences = load_frame_ids(args.split)
    print(f"VoD {args.split}: {len(frame_ids)} frames, {len(sequences)} sequences")

    # Preload data (once)
    preloaded = preload_gt_data(sequences)

    def run_one(name, doppler_cfg):
        cfg = copy.deepcopy(base_cfg)
        cfg['doppler'] = doppler_cfg
        print(f"  Running: {name}...", end=' ', flush=True)
        t0 = time.time()
        pred = run_tracker(cfg, sequences, preloaded)
        tracking_time = time.time() - t0
        metrics = evaluate(gt_by_frame, pred, frame_ids)
        fm = format_metrics(metrics)
        mota = fm['OVERALL']['mota']
        ids = fm['OVERALL']['ids']
        print(f"done ({tracking_time:.1f}s) MOTA={mota:.3f} IDS={ids:.0f}")
        return fm, tracking_time

    # ============================================================
    # PART 1: Core 6-row ablation
    # ============================================================
    if not args.only_sensitivity:
        print(f"\n{'='*80}")
        print("  PART 1: Core Ablation (6 configs, oracle GT boxes)")
        print(f"{'='*80}")

        ablation_configs = [
            ('Baseline',     dict(C=False, D=False)),
            ('C only',       dict(C=True,  D=False)),
            ('D only',       dict(C=False, D=True)),
            ('C+D',          dict(C=True,  D=True)),
            ('D nofilter',   dict(C=False, D=True,  vr_class_thre=ZERO_VR_THRE)),
            ('CD nofilter',  dict(C=True,  D=True,  vr_class_thre=ZERO_VR_THRE)),
            # AN and RCS ablation
            ('AN only',      dict(AN=True)),
            ('RCS only',     dict(RCS=True)),
            ('+AN',          dict(C=True,  D=True,  AN=True, vr_class_thre=ZERO_VR_THRE)),
            ('+RCS',         dict(C=True,  D=True,  RCS=True, vr_class_thre=ZERO_VR_THRE)),
            ('Full (CDRA)',  dict(C=True,  D=True,  AN=True, RCS=True, vr_class_thre=ZERO_VR_THRE)),
        ]

        ablation_results = {}
        ablation_times = {}
        for name, kwargs in ablation_configs:
            doppler_cfg = make_doppler_cfg(**kwargs)
            fm, t = run_one(name, doppler_cfg)
            ablation_results[name] = fm
            ablation_times[name] = t

        print_table(ablation_results)

        # Runtime comparison
        print(f"\n  Runtime (tracking only):")
        for name in ['Baseline', 'C+D']:
            if name in ablation_times:
                n_frames = sum(len(seq) for seq in sequences)
                fps = n_frames / ablation_times[name]
                print(f"    {name}: {ablation_times[name]:.1f}s total, "
                      f"{1000*ablation_times[name]/n_frames:.1f}ms/frame, {fps:.1f} FPS")

    # ============================================================
    # PART 2: Threshold sensitivity
    # ============================================================
    if not args.only_ablation:
        print(f"\n{'='*80}")
        print("  PART 2: Threshold Sensitivity (C+D, varying filter thresholds)")
        print(f"{'='*80}")

        factors = [0.5, 1.0, 1.5, 2.0]
        sensitivity_results = {}
        for factor in factors:
            scaled_thre = {k: v * factor for k, v in FINAL_VR_THRE.items()}
            name = f"x{factor:.1f}"
            thre_str = ", ".join([f"{CLASS_NAMES[k]}={v:.2f}" for k, v in scaled_thre.items()])
            print(f"  Factor {name}: {thre_str}")
            doppler_cfg = make_doppler_cfg(C=True, D=True, vr_class_thre=scaled_thre)
            fm, t = run_one(name, doppler_cfg)
            sensitivity_results[name] = fm

        print_table(sensitivity_results, baseline_key='x1.0')

    print("\nAll done!")


if __name__ == '__main__':
    main()
