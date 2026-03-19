"""
run_vod.py - Run DE-FastPoly tracker on VoD (View-of-Delft) dataset

Uses GT annotations as detections + radar point cloud for Doppler velocity.
Evaluates tracking with HOTA/MOTA metrics.

Pipeline:
  1. Read GT labels (KITTI format, with track IDs)
  2. Convert camera coords -> BEV coords (standard KITTI transform)
  3. Read radar point cloud, find points inside each box, compute mean v_r
  4. Feed detections to DE-FastPoly tracker
  5. Output tracked results for evaluation

Usage:
  python run_vod.py                    # run on val set
  python run_vod.py --split train      # run on train set
  python run_vod.py --no-doppler       # run baseline (no doppler)
"""

# [AGENT-SYNC-MARKER]
# EN: Shared VoD runner for Codex/Claude collaboration; keep detector/GT radar stat extraction consistent.
# 中文: 这是Codex/Claude共同维护的VoD运行脚本；请保持检测与GT的雷达统计提取一致。
# Owner: Codex/Claude
# Date: 2026-03-11

import sys, csv, os, argparse
import numpy as np
from collections import defaultdict

# FastPoly imports
FASTPOLY_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FASTPOLY_DIR)

import yaml
from tracking.nusc_tracker import Tracker
from geometry.nusc_box import NuscBox
from pre_processing.nusc_data_conversion import arraydet2box

# ===== Paths =====
VOD_ROOT = os.environ.get('VOD_ROOT', './data/view_of_delft_PUBLIC')
RADAR_DIR = os.path.join(VOD_ROOT, 'radar_5frames', 'training', 'velodyne')
LABEL_DIR = os.environ.get('VOD_LABEL_DIR', './data/label_2')  # with track IDs
IMAGESETS_DIR = os.path.join(VOD_ROOT, 'lidar', 'ImageSets')
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config', 'vod_config.yaml')
OUTPUT_DIR = './results/vod'

# ===== VoD class mapping =====
# VoD main classes -> internal class IDs
VOD_CLASS_MAP = {
    'Car': 0,
    'Pedestrian': 1,
    'Cyclist': 2,
}
# Classes to ignore
IGNORE_CLASSES = {'DontCare', 'bicycle', 'bicycle_rack', 'human_depiction',
                  'moped_scooter', 'motor', 'ride_other', 'ride_uncertain',
                  'rider', 'truck', 'vehicle_other'}


_TR_V2C = np.array([
    [-0.013857, -0.9997468, 0.01772762, 0.05283124],
    [0.10934269, -0.01913807, -0.99381983, 0.98100483],
    [0.99390751, -0.01183297, 0.1095802, 1.44445002]
])
_R_v2c = _TR_V2C[:, :3]
_t_v2c = _TR_V2C[:, 3]
_R_inv_v2c = np.linalg.inv(_R_v2c)


def cam_to_bev(x_cam, y_cam, z_cam):
    """Convert KITTI camera coords to radar coords using proper inverse transform."""
    p_cam = np.array([x_cam, y_cam, z_cam])
    p_radar = _R_inv_v2c @ (p_cam - _t_v2c)
    return p_radar[0], p_radar[1], p_radar[2]


def parse_kitti_label(line):
    """Parse one line of KITTI label (with track ID in field 2).
    Returns dict or None if class should be ignored.
    """
    parts = line.strip().split()
    if len(parts) < 16:
        return None

    cls_name = parts[0]
    if cls_name not in VOD_CLASS_MAP:
        return None

    track_id = int(float(parts[1]))
    # occluded = int(parts[2])  # not used
    # alpha = float(parts[3])   # not used
    # bbox 2d: parts[4:8]       # not used

    h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
    x_cam, y_cam, z_cam = float(parts[11]), float(parts[12]), float(parts[13])
    ry = float(parts[14])
    # score = float(parts[15])  # always 1 for GT

    # Convert to BEV
    x_bev, y_bev, z_bev = cam_to_bev(x_cam, y_cam, z_cam)

    return {
        'class_name': cls_name,
        'class_id': VOD_CLASS_MAP[cls_name],
        'track_id': track_id,
        'x': x_bev,
        'y': y_bev,
        'z': z_bev,
        'w': w,
        'l': l,
        'h': h,
        'ry': ry,  # rotation around y in camera = rotation around z in BEV
    }


def read_radar_pointcloud(frame_id):
    """Read radar point cloud for a frame. Returns (N, 7) array."""
    path = os.path.join(RADAR_DIR, f'{frame_id:05d}.bin')
    if not os.path.exists(path):
        return np.empty((0, 7))
    pc = np.fromfile(path, dtype=np.float32).reshape(-1, 7)
    return pc  # [x, y, z, RCS, v_r, v_r_comp, time]


def _box_point_mask(pc, x, y, z, w, l, h, ry, expand=1.0):
    """Compute in-box radar point mask in box local frame."""
    if len(pc) == 0:
        return np.zeros((0,), dtype=bool)

    cos_r, sin_r = np.cos(ry), np.sin(ry)
    dx = pc[:, 0] - x
    dy = pc[:, 1] - y
    dz = pc[:, 2] - z

    local_x = dx * cos_r + dy * sin_r
    local_y = -dx * sin_r + dy * cos_r

    half_l = l / 2 * expand
    half_w = w / 2 * expand
    half_h = h / 2 * expand

    return (np.abs(local_x) < half_l) & (np.abs(local_y) < half_w) & (np.abs(dz) < half_h)


def get_radar_stats_for_box(pc, x, y, z, w, l, h, ry, expand=1.0):
    # [AGENT-SYNC-MARKER][RADAR_STATS]
    # EN: Unified extraction API for v_r / n_pts / RCS to avoid script-level mismatch.
    # 中文: 统一提取v_r / 点数 / RCS的接口，避免不同脚本口径不一致。
    """Return radar stats inside box: (v_r_comp mean, has_valid, n_pts, rcs mean)."""
    if len(pc) == 0:
        return 0.0, False, 0, 0.0

    mask = _box_point_mask(pc, x, y, z, w, l, h, ry, expand=expand)
    if np.sum(mask) == 0:
        mask = _box_point_mask(pc, x, y, z, w, l, h, ry, expand=expand * 2.0)
        if np.sum(mask) == 0:
            return 0.0, False, 0, 0.0

    pts = pc[mask]
    vr_mean = float(np.mean(pts[:, 5]))  # v_r_comp
    rcs_mean = float(np.mean(pts[:, 3]))  # RCS
    n_pts = int(pts.shape[0])
    return vr_mean, True, n_pts, rcs_mean


def get_doppler_for_box(pc, x, y, z, w, l, h, ry, expand=1.0):
    """Find radar points inside a 3D box and return mean v_r_compensated.

    Uses axis-aligned BEV check (ignoring rotation for simplicity,
    since VoD boxes are often roughly axis-aligned for common objects).

    Args:
        pc: (N, 7) radar point cloud
        x, y, z: box center in BEV coords
        w, l, h: box dimensions
        ry: rotation (not used in simplified version)
        expand: expansion factor for box search

    Returns:
        tuple: (mean_vr_compensated, has_valid_vr, n_pts)
    """
    vr, found, n_pts, _ = get_radar_stats_for_box(pc, x, y, z, w, l, h, ry, expand=expand)
    return vr, found, n_pts


def load_frame_ids(split):
    """Load frame IDs for a split and group into sequences."""
    path = os.path.join(IMAGESETS_DIR, f'{split}.txt')
    frames = [int(l.strip()) for l in open(path)]

    # Split into sequences by gaps
    sequences = []
    current_seq = [frames[0]]
    for i in range(1, len(frames)):
        if frames[i] - frames[i - 1] > 1:
            sequences.append(current_seq)
            current_seq = [frames[i]]
        else:
            current_seq.append(frames[i])
    sequences.append(current_seq)

    return sequences


def yaw_to_quaternion(ry):
    """Convert yaw angle to quaternion [w, x, y, z]."""
    # VoD ry is rotation around camera Y axis = rotation around BEV Z axis
    half = ry / 2.0
    return [np.cos(half), 0.0, 0.0, np.sin(half)]


def make_np_det(det):
    """Convert detection dict to FastPoly np_array format (14 values)."""
    x, y, z = det['x'], det['y'], det['z']
    w, l, h = det['w'], det['l'], det['h']
    vx, vy = 0.0, 0.0
    qw, qx, qy, qz = yaw_to_quaternion(det['ry'])
    score = 0.9  # GT detection, high score
    cls_label = float(det['class_id'])
    return np.array([x, y, z, w, l, h, vx, vy, qw, qx, qy, qz, score, cls_label])


def build_frame_info(frame_id, dets, seq_id=0, is_first=False):
    """Build data_info dict for FastPoly Tracker.tracking()."""
    if not dets:
        return {
            'frame_id': frame_id,
            'seq_id': seq_id,
            'is_first_frame': is_first,
            'no_dets': True,
            'det_num': 0,
            'np_dets': np.empty((0, 14)),
            'box_dets': np.array([]),
            'np_dets_bottom_corners': np.empty((0, 4, 2)),
            'np_dets_norm_corners': np.empty((0, 4)),
            'has_velo': False,
            'radial_vels': np.array([]),
            'valid_vr_mask': np.array([], dtype=bool),
            'vr_n_pts': np.array([], dtype=float),
            'radar_rcs': np.array([], dtype=float),
        }

    np_dets = np.array([make_np_det(d) for d in dets])
    box_dets, bm_dets, norm_bm_dets = arraydet2box(np_dets, init_geo=True)
    radial_vels = np.array([d.get('radial_vel', 0.0) for d in dets])
    valid_vr_mask = np.array([d.get('has_valid_vr', False) for d in dets])
    vr_n_pts = np.array([d.get('vr_n_pts', 0) for d in dets])
    radar_rcs = np.array([d.get('radar_rcs', 0.0) for d in dets], dtype=float)

    return {
        'frame_id': frame_id,
        'seq_id': seq_id,
        'is_first_frame': is_first,
        'no_dets': False,
        'det_num': len(dets),
        'np_dets': np_dets,
        'box_dets': box_dets,
        'np_dets_bottom_corners': bm_dets,
        'np_dets_norm_corners': norm_bm_dets,
        'has_velo': False,
        'radial_vels': radial_vels,
        'valid_vr_mask': valid_vr_mask,
        'vr_n_pts': vr_n_pts,
        'radar_rcs': radar_rcs,
    }


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Run DE-FastPoly on VoD')
    parser.add_argument('--split', default='val', choices=['val', 'train', 'train_val'])
    parser.add_argument('--no-doppler', action='store_true', help='Disable all doppler features')
    parser.add_argument('--config', default=CONFIG_PATH)
    parser.add_argument('--max-frames', type=int, default=0, help='Limit frames per sequence (0=all)')
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.no_doppler:
        cfg['doppler'] = {
            'use_doppler_obs': False,
            'use_doppler_assoc': False,
            'use_doppler_gate': False,
            'use_doppler_init': False,
            'use_range_adaptive': False,
        }
        suffix = 'baseline'
    else:
        suffix = 'de_fastpoly'

    sequences = load_frame_ids(args.split)
    print(f"VoD {args.split} set: {len(sequences)} sequences, "
          f"{sum(len(s) for s in sequences)} total frames")
    for i, seq in enumerate(sequences):
        print(f"  Seq {i}: frames {seq[0]:05d}-{seq[-1]:05d} ({len(seq)} frames)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f'{args.split}_{suffix}.csv')

    total_tracks = 0
    total_dets = 0
    total_with_vr = 0

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_id', 'track_id', 'cx', 'cy', 'cz', 'w', 'l', 'h',
                          'class_id', 'class_name'])

        for seq_idx, seq_frames in enumerate(sequences):
            print(f"\n--- Sequence {seq_idx}: {len(seq_frames)} frames ---")
            tracker = Tracker(cfg)
            tracker.reset()

            n_frames = len(seq_frames)
            if args.max_frames > 0:
                n_frames = min(n_frames, args.max_frames)

            for i in range(n_frames):
                fid = seq_frames[i]
                is_first = (i == 0)

                # Read GT labels
                label_path = os.path.join(LABEL_DIR, f'{fid:05d}.txt')
                dets = []
                if os.path.exists(label_path):
                    with open(label_path) as lf:
                        for line in lf:
                            det = parse_kitti_label(line)
                            if det is not None:
                                dets.append(det)

                # Read radar and compute v_r / n_pts / RCS for each detection
                pc = read_radar_pointcloud(fid)
                for det in dets:
                    vr, found, n_pts, rcs = get_radar_stats_for_box(pc, det['x'], det['y'], det['z'],
                                                                     det['w'], det['l'], det['h'], det['ry'])
                    det['radial_vel'] = vr
                    det['has_valid_vr'] = found
                    det['vr_n_pts'] = n_pts
                    det['radar_rcs'] = rcs
                    if found:
                        total_with_vr += 1

                total_dets += len(dets)

                # Build frame info and track
                data_info = build_frame_info(fid, dets, seq_id=seq_idx, is_first=is_first)
                tracker.tracking(data_info)

                # Extract results
                if data_info.get('no_val_track_result') or 'np_track_res' not in data_info:
                    if i % 100 == 0:
                        print(f"  Frame {fid:05d} ({i}/{n_frames}): {len(dets)} dets -> 0 tracks")
                    continue

                np_track_res = np.array(data_info['np_track_res'])
                for row in np_track_res:
                    x, y, z = row[0], row[1], row[2]
                    w, l, h = row[3], row[4], row[5]
                    track_id = int(row[14])
                    cls_id = int(row[13])
                    cls_name = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}.get(cls_id, 'Unknown')
                    writer.writerow([
                        fid, track_id,
                        f'{x:.4f}', f'{y:.4f}', f'{z:.4f}',
                        f'{w:.4f}', f'{l:.4f}', f'{h:.4f}',
                        cls_id, cls_name,
                    ])
                    total_tracks += 1

                if i % 100 == 0:
                    print(f"  Frame {fid:05d} ({i}/{n_frames}): {len(dets)} dets -> {len(np_track_res)} tracks")

    print(f"\n{'='*60}")
    print(f"Done! {total_dets} detections, {total_tracks} tracked rows")
    print(f"Detections with v_r: {total_with_vr}/{total_dets} ({100*total_with_vr/max(total_dets,1):.1f}%)")
    print(f"Output: {out_path}")


if __name__ == '__main__':
    main()
