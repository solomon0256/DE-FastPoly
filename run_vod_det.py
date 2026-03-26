"""
run_vod_det.py - Run DE-FastPoly tracker on VoD using PointPillars detections

Uses real PointPillars detector output (not GT) + radar point cloud for Doppler.
Every detection inherently has radar points -> valid v_r -> Doppler contributions work.

Usage:
  python run_vod_det.py                    # DE-FastPoly (all doppler on)
  python run_vod_det.py --no-doppler       # baseline (no doppler)
  python run_vod_det.py --adaptive-noise   # enable Radar-Informed Adaptive Noise
"""

import sys, csv, os, argparse
import numpy as np
from collections import defaultdict

# FastPoly imports
FASTPOLY_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FASTPOLY_DIR)

import yaml
from tracking.nusc_tracker import Tracker
from pre_processing.nusc_data_conversion import arraydet2box

# ===== Paths =====
VOD_ROOT = os.environ.get('VOD_ROOT', './data/view_of_delft_PUBLIC')
RADAR_DIR = os.path.join(VOD_ROOT, 'radar_5frames', 'training', 'velodyne')
IMAGESETS_DIR = os.path.join(VOD_ROOT, 'lidar', 'ImageSets')
DET_DIR = './results/vod_detections'  # PointPillars output
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'vod_config.yaml')
OUTPUT_DIR = './results/vod'

VOD_CLASS_MAP = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

# VoD radar Tr_velo_to_cam (radar -> camera transform)
TR_VELO_TO_CAM = np.array([
    [-0.013857, -0.9997468, 0.01772762, 0.05283124],
    [0.10934269, -0.01913807, -0.99381983, 0.98100483],
    [0.99390751, -0.01183297, 0.1095802, 1.44445002]
])
_R = TR_VELO_TO_CAM[:, :3]
_t = TR_VELO_TO_CAM[:, 3]
_R_inv = np.linalg.inv(_R)


def cam_to_radar(x_cam, y_cam, z_cam):
    """Convert camera coords to radar coords using proper inverse transform."""
    p_cam = np.array([x_cam, y_cam, z_cam])
    p_radar = _R_inv @ (p_cam - _t)
    return p_radar[0], p_radar[1], p_radar[2]


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_vod import (read_radar_pointcloud, get_radar_stats_for_box,
                     build_frame_info, load_frame_ids)
from eval_vod import load_gt, evaluate, print_metrics, find_moving_track_ids


def parse_det_label(line, score_thre=0.1):
    """Parse one line of PointPillars detection output (KITTI format).
    Format: class -1 -1 alpha x1 y1 x2 y2 h w l x y z ry score
    """
    parts = line.strip().split()
    if len(parts) < 16:
        return None

    cls_name = parts[0]
    if cls_name not in VOD_CLASS_MAP:
        return None

    score = float(parts[15])
    if score < score_thre:
        return None

    h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
    x_cam, y_cam, z_cam = float(parts[11]), float(parts[12]), float(parts[13])
    ry = float(parts[14])

    # Now cam_to_bev in run_vod uses proper inverse -> radar coords
    from run_vod import cam_to_bev
    x_bev, y_bev, z_bev = cam_to_bev(x_cam, y_cam, z_cam)

    return {
        'class_name': cls_name,
        'class_id': VOD_CLASS_MAP[cls_name],
        'x': x_bev, 'y': y_bev, 'z': z_bev,
        'w': w, 'l': l, 'h': h,
        'ry': ry,
        'score': score,
    }


def make_np_det_scored(det):
    """Convert detection dict to FastPoly np_array format (14 values)."""
    x, y, z = det['x'], det['y'], det['z']
    w, l, h = det['w'], det['l'], det['h']
    vx, vy = 0.0, 0.0
    half = det['ry'] / 2.0
    qw, qx, qy, qz = np.cos(half), 0.0, 0.0, np.sin(half)
    score = det['score']
    cls_label = float(det['class_id'])
    return np.array([x, y, z, w, l, h, vx, vy, qw, qx, qy, qz, score, cls_label])


def main():
    parser = argparse.ArgumentParser(description='Run DE-FastPoly on VoD with PointPillars detections')
    parser.add_argument('--split', default='val')
    parser.add_argument('--no-doppler', action='store_true')
    parser.add_argument('--adaptive-noise', action='store_true',
                        help='Enable Radar-Informed Adaptive Noise in KF')
    parser.add_argument('--score-thre', type=float, default=0.2)
    parser.add_argument('--config', default=CONFIG_PATH)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding='utf-8'))

    if args.no_doppler:
        cfg['doppler'] = {
            'use_doppler_obs': False, 'use_doppler_assoc': False,
            'use_doppler_gate': False, 'use_doppler_init': False,
            'use_range_adaptive': False,
            'use_adaptive_noise': False,
        }
        suffix = 'det_baseline'
    else:
        suffix = 'det_de_fastpoly'

    if args.adaptive_noise:
        cfg.setdefault('doppler', {})
        cfg['doppler'].update({
            'use_adaptive_noise': True,
            'adapt_range_ref': 30.0,
            'adapt_npts_ref': 5.0,
            'adapt_k_range': 0.25,
            'adapt_k_score': 0.35,
            'adapt_k_vr': 0.20,
            'adapt_k_npts': 0.15,
            'adapt_min_scale': 0.5,
            'adapt_max_scale': 2.5,
        })
        suffix += '_adaptR'

    sequences = load_frame_ids(args.split)
    total_frames = sum(len(s) for s in sequences)
    print(f"VoD {args.split}: {len(sequences)} seqs, {total_frames} frames")

    # Load GT for evaluation
    gt_by_frame, frame_ids = load_gt(args.split)
    total_gt = sum(len(v) for v in gt_by_frame.values())
    print(f"GT: {total_gt} objects across {len(frame_ids)} frames")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f'{args.split}_{suffix}.csv')

    pred_by_frame = defaultdict(list)
    total_dets = 0
    total_with_vr = 0
    total_tracks = 0

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_id', 'track_id', 'cx', 'cy', 'cz', 'w', 'l', 'h',
                          'class_id', 'class_name', 'score'])

        for seq_idx, seq_frames in enumerate(sequences):
            tracker = Tracker(cfg)
            tracker.reset()

            for i, fid in enumerate(seq_frames):
                is_first = (i == 0)

                # Read PointPillars detections
                det_path = os.path.join(DET_DIR, f'{fid:05d}.txt')
                dets = []
                if os.path.exists(det_path):
                    with open(det_path) as lf:
                        for line in lf:
                            det = parse_det_label(line, args.score_thre)
                            if det is not None:
                                dets.append(det)

                # Read radar and compute v_r / n_pts / RCS (coords now in radar frame)
                pc = read_radar_pointcloud(fid)
                for det in dets:
                    vr, found, n_pts, rcs = get_radar_stats_for_box(pc, det['x'], det['y'], det['z'],
                                                                     det['w'], det['l'], det['h'], det['ry'])
                    det['radial_vel'] = vr
                    det['has_valid_vr'] = found  # True = radar points found in box
                    det['vr_n_pts'] = n_pts
                    det['radar_rcs'] = rcs
                    if found:
                        total_with_vr += 1

                total_dets += len(dets)

                # Build frame info
                if not dets:
                    data_info = build_frame_info(fid, dets, seq_id=seq_idx, is_first=is_first)
                else:
                    np_dets = np.array([make_np_det_scored(d) for d in dets])
                    box_dets, bm_dets, norm_bm_dets = arraydet2box(np_dets, init_geo=True)
                    radial_vels = np.array([d.get('radial_vel', 0.0) for d in dets])
                    valid_vr_mask = np.array([d.get('has_valid_vr', False) for d in dets])
                    vr_n_pts = np.array([d.get('vr_n_pts', 0) for d in dets])
                    radar_rcs = np.array([d.get('radar_rcs', 0.0) for d in dets], dtype=float)

                    data_info = {
                        'frame_id': fid, 'seq_id': seq_idx,
                        'is_first_frame': is_first, 'no_dets': False,
                        'det_num': len(dets), 'np_dets': np_dets,
                        'box_dets': box_dets,
                        'np_dets_bottom_corners': bm_dets,
                        'np_dets_norm_corners': norm_bm_dets,
                        'has_velo': False,
                        'radial_vels': radial_vels,
                        'valid_vr_mask': valid_vr_mask,
                        'vr_n_pts': vr_n_pts,
                        'radar_rcs': radar_rcs,
                    }

                tracker.tracking(data_info)

                # Extract results
                if data_info.get('no_val_track_result') or 'np_track_res' not in data_info:
                    continue

                np_track_res = np.array(data_info['np_track_res'])
                for row in np_track_res:
                    track_id = int(row[14])
                    cls_id = int(row[13])
                    cls_name = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}.get(cls_id, 'Unknown')
                    writer.writerow([
                        fid, track_id,
                        f'{row[0]:.4f}', f'{row[1]:.4f}', f'{row[2]:.4f}',
                        f'{row[3]:.4f}', f'{row[4]:.4f}', f'{row[5]:.4f}',
                        cls_id, cls_name, f'{row[12]:.4f}',
                    ])
                    pred_by_frame[fid].append({
                        'track_id': track_id, 'class_id': cls_id,
                        'x': row[0], 'y': row[1], 'z': row[2],
                    })
                    total_tracks += 1

            print(f"  Seq {seq_idx}: {len(seq_frames)} frames done")

    print(f"\n{'='*60}")
    print(f"Detections: {total_dets} total, {total_with_vr} with v_r ({100*total_with_vr/max(total_dets,1):.1f}%)")
    print(f"Tracked: {total_tracks} rows -> {out_path}")

    # Evaluate - all objects with class-specific thresholds
    metrics = evaluate(gt_by_frame, pred_by_frame, frame_ids)
    print_metrics(metrics, label=f"{suffix} [ALL, class-specific dist]")

    # Evaluate - moving objects only
    moving_ids = find_moving_track_ids(gt_by_frame, frame_ids, move_thre=1.0)
    metrics_mov = evaluate(gt_by_frame, pred_by_frame, frame_ids,
                           moving_only=True, moving_ids=moving_ids)
    print_metrics(metrics_mov, label=f"{suffix} [MOVING only]")


if __name__ == '__main__':
    main()
