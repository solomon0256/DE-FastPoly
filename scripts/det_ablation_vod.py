"""
det_ablation_vod.py - Ablation on VoD with real PointPillars detections

Runs 4 configs: Baseline / C only / D only / C+D (with threshold).
Reports ALL + MOVING-only metrics + per-class IDS.

Usage:
  python det_ablation_vod.py
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
from run_vod import read_radar_pointcloud, get_radar_stats_for_box, build_frame_info, load_frame_ids
from run_vod_det import parse_det_label, make_np_det_scored, DET_DIR
from eval_vod import load_gt, evaluate, find_moving_track_ids

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'vod_config.yaml')
CLASS_NAMES = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}

# Same parameters as paper_ablation.py
FINAL_ALPHA = 0.95
FINAL_BETA = 0.05
FINAL_SIGMA = 10.0
FINAL_VR_THRE = {0: 1.0, 1: 0.3, 2: 0.5}


def make_doppler_cfg(C=False, D=False, vr_class_thre=None):
    """Build doppler config. Same as paper_ablation.py."""
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
    }
    if vr_class_thre is not None:
        cfg['vr_class_thre'] = vr_class_thre
    else:
        cfg['vr_class_thre'] = FINAL_VR_THRE.copy()
    return cfg


def preload_det_data(sequences, score_thre=0.2):
    """Preload PointPillars detections + radar v_r for all frames."""
    all_fids = [fid for seq in sequences for fid in seq]
    print(f"Preloading {len(all_fids)} frames (PointPillars detections)...")
    t0 = time.time()

    preloaded = {}
    total_dets = 0
    total_vr = 0

    for fid in all_fids:
        # Read PointPillars detections
        det_path = os.path.join(DET_DIR, f'{fid:05d}.txt')
        dets = []
        if os.path.exists(det_path):
            with open(det_path) as lf:
                for line in lf:
                    det = parse_det_label(line, score_thre)
                    if det is not None:
                        dets.append(det)

        # Read radar and compute v_r / n_pts / RCS
        pc = read_radar_pointcloud(fid)
        for det in dets:
            vr, found, n_pts, rcs = get_radar_stats_for_box(pc, det['x'], det['y'], det['z'],
                                                            det['w'], det['l'], det['h'], det['ry'])
            det['radial_vel'] = vr
            det['has_valid_vr'] = found
            det['vr_n_pts'] = n_pts
            det['radar_rcs'] = rcs
            if found:
                total_vr += 1
        total_dets += len(dets)

        entry = {'dets': dets}
        if dets:
            np_dets = np.array([make_np_det_scored(d) for d in dets])
            box_dets, bm_dets, norm_bm_dets = arraydet2box(np_dets, init_geo=True)
            entry['np_dets'] = np_dets
            entry['box_dets'] = box_dets
            entry['bm_dets'] = bm_dets
            entry['norm_bm_dets'] = norm_bm_dets
            entry['radial_vels'] = np.array([d['radial_vel'] for d in dets])
            entry['valid_vr_mask'] = np.array([d['has_valid_vr'] for d in dets])
            entry['vr_n_pts'] = np.array([d.get('vr_n_pts', 0) for d in dets])
            entry['radar_rcs'] = np.array([d.get('radar_rcs', 0.0) for d in dets], dtype=float)
        preloaded[fid] = entry

    elapsed = time.time() - t0
    print(f"Preloaded in {elapsed:.1f}s: {total_dets} detections, "
          f"{total_vr} with v_r ({100*total_vr/max(total_dets,1):.1f}%)")
    return preloaded


def run_tracker(cfg, sequences, preloaded):
    """Run tracker on preloaded data, return pred_by_frame."""
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
                    'vr_n_pts': preloaded[fid].get('vr_n_pts', np.array([])),
                    'radar_rcs': preloaded[fid].get('radar_rcs', np.array([])),
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
    """Extract formatted metrics dict."""
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
            'n_gt': m['n_gt'],
        }
    return result


def print_table(all_results, label, baseline_key='Baseline'):
    """Print formatted ablation table."""
    bl = all_results.get(baseline_key, {}).get('OVERALL', {})

    print(f"\n{'='*110}")
    print(f"  {label}")
    print(f"{'='*110}")
    print(f"  {'Config':<20s} {'MOTA':>7} {'MOTP':>7} {'IDS':>5} "
          f"{'TP':>6} {'FP':>6} {'FN':>6}  "
          f"{'Car':>7} {'Ped':>7} {'Cyc':>7}  {'dMOTA':>7} {'dIDS':>5}")
    print(f"  {'-'*105}")

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

    print(f"{'='*110}")

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
    base_cfg = yaml.safe_load(open(CONFIG_PATH, encoding='utf-8'))

    # Load GT and sequences
    gt_by_frame, frame_ids = load_gt('val')
    moving_ids = find_moving_track_ids(gt_by_frame, frame_ids, move_thre=1.0)
    sequences = load_frame_ids('val')
    n_frames = sum(len(seq) for seq in sequences)
    print(f"VoD val: {len(frame_ids)} frames, {len(sequences)} sequences")

    # Preload detections (once)
    preloaded = preload_det_data(sequences)

    # 4 ablation configs (with threshold, NOT nofilter)
    ablation_configs = [
        ('Baseline', dict(C=False, D=False)),
        ('C only',   dict(C=True,  D=False)),
        ('D only',   dict(C=False, D=True)),
        ('C+D',      dict(C=True,  D=True)),
    ]

    all_results = {}
    all_results_moving = {}

    for name, kwargs in ablation_configs:
        doppler_cfg = make_doppler_cfg(**kwargs)
        cfg = copy.deepcopy(base_cfg)
        cfg['doppler'] = doppler_cfg

        print(f"\n  Running: {name}...", end=' ', flush=True)
        t0 = time.time()
        pred = run_tracker(cfg, sequences, preloaded)
        elapsed = time.time() - t0

        # ALL objects
        metrics = evaluate(gt_by_frame, pred, frame_ids)
        fm = format_metrics(metrics)
        all_results[name] = fm
        mota = fm['OVERALL']['mota']
        ids = fm['OVERALL']['ids']

        # MOVING only
        metrics_mov = evaluate(gt_by_frame, pred, frame_ids,
                               moving_only=True, moving_ids=moving_ids)
        fm_mov = format_metrics(metrics_mov)
        all_results_moving[name] = fm_mov
        mota_mov = fm_mov['OVERALL']['mota']
        ids_mov = fm_mov['OVERALL']['ids']

        fps = n_frames / elapsed
        print(f"done ({elapsed:.1f}s, {fps:.1f} FPS) "
              f"ALL: MOTA={mota:.3f} IDS={ids:.0f} | "
              f"MOV: MOTA={mota_mov:.3f} IDS={ids_mov:.0f}")

    # Print tables
    print_table(all_results, "PointPillars Detection — ALL objects")
    print_table(all_results_moving, "PointPillars Detection — MOVING only")

    print("\nDone!")


if __name__ == '__main__':
    main()
