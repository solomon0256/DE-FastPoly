"""
eval_vod.py - Evaluate tracking results on VoD using GT track IDs

Computes: MOTA, MOTP, IDS (ID Switches), FP, FN, Precision, Recall
Supports:
  - Class-specific distance thresholds (Car: 3m, Ped: 1.5m, Cyc: 2m)
  - Moving-only evaluation (filter out static GT objects)
  - Per-class and overall metrics

Usage:
  python eval_vod.py results/vod/val_baseline.csv results/vod/val_de_fastpoly.csv
"""
import sys, os, csv
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

# VoD paths
VOD_ROOT = os.environ.get('VOD_ROOT', './data/view_of_delft_PUBLIC')
LABEL_DIR = os.environ.get('VOD_LABEL_DIR', './data/label_2')  # with track IDs
IMAGESETS_DIR = os.path.join(VOD_ROOT, 'lidar', 'ImageSets')

VOD_CLASS_MAP = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
CLASS_NAMES = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}
IGNORE_CLASSES = {'DontCare', 'bicycle', 'bicycle_rack', 'human_depiction',
                  'moped_scooter', 'motor', 'ride_other', 'ride_uncertain',
                  'rider', 'truck', 'vehicle_other'}

# Class-specific distance thresholds (meters) for matching
# Roughly based on half the typical object size
CLASS_DIST_THRE = {0: 3.0, 1: 1.5, 2: 2.0}  # Car, Ped, Cyclist

# VoD radar Tr_velo_to_cam
_TR = np.array([
    [-0.013857, -0.9997468, 0.01772762, 0.05283124],
    [0.10934269, -0.01913807, -0.99381983, 0.98100483],
    [0.99390751, -0.01183297, 0.1095802, 1.44445002]
])
_R_eval = _TR[:, :3]
_t_eval = _TR[:, 3]
_R_inv_eval = np.linalg.inv(_R_eval)


def cam_to_bev(x_cam, y_cam, z_cam):
    """Convert camera coords to radar coords using proper inverse transform."""
    p_cam = np.array([x_cam, y_cam, z_cam])
    p_radar = _R_inv_eval @ (p_cam - _t_eval)
    return p_radar[0], p_radar[1], p_radar[2]


def load_gt(split='val'):
    """Load GT annotations grouped by frame. Returns {frame_id: [gt_objects]}."""
    split_path = os.path.join(IMAGESETS_DIR, f'{split}.txt')
    frame_ids = [int(l.strip()) for l in open(split_path)]

    gt_by_frame = {}
    for fid in frame_ids:
        label_path = os.path.join(LABEL_DIR, f'{fid:05d}.txt')
        objects = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 16:
                        continue
                    cls_name = parts[0]
                    if cls_name not in VOD_CLASS_MAP:
                        continue
                    track_id = int(float(parts[1]))
                    x_cam, y_cam, z_cam = float(parts[11]), float(parts[12]), float(parts[13])
                    x, y, z = cam_to_bev(x_cam, y_cam, z_cam)
                    objects.append({
                        'track_id': track_id,
                        'class_id': VOD_CLASS_MAP[cls_name],
                        'class_name': cls_name,
                        'x': x, 'y': y, 'z': z,
                    })
        gt_by_frame[fid] = objects

    return gt_by_frame, frame_ids


def find_moving_track_ids(gt_by_frame, frame_ids, move_thre=1.0):
    """Find GT track IDs that move more than move_thre meters across their lifetime."""
    track_positions = defaultdict(list)  # track_id -> [(x, y)]
    for fid in frame_ids:
        for obj in gt_by_frame.get(fid, []):
            track_positions[obj['track_id']].append((obj['x'], obj['y']))

    moving_ids = set()
    for tid, positions in track_positions.items():
        if len(positions) < 2:
            continue
        positions = np.array(positions)
        dx = positions[:, 0].max() - positions[:, 0].min()
        dy = positions[:, 1].max() - positions[:, 1].min()
        total_displacement = np.sqrt(dx**2 + dy**2)
        if total_displacement > move_thre:
            moving_ids.add(tid)

    return moving_ids


def load_predictions(csv_path):
    """Load tracked predictions grouped by frame."""
    pred_by_frame = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            fid = int(row['frame_id'])
            pred_by_frame[fid].append({
                'track_id': int(row['track_id']),
                'class_id': int(row['class_id']),
                'x': float(row['cx']),
                'y': float(row['cy']),
                'z': float(row['cz']),
            })
    return pred_by_frame


def evaluate(gt_by_frame, pred_by_frame, frame_ids, dist_thre=None,
             moving_only=False, moving_ids=None, matching='greedy'):
    """Compute CLEAR MOT metrics (MOTA, MOTP, IDS, FP, FN).

    Args:
        dist_thre: float or None. If None, use class-specific thresholds.
        moving_only: if True, only evaluate GT objects whose track_id is in moving_ids
        moving_ids: set of track IDs that are considered moving
    """
    # Per-class accumulators
    metrics = {}
    for cls_id in [None, 0, 1, 2]:  # None = overall
        metrics[cls_id] = {
            'tp': 0, 'fp': 0, 'fn': 0, 'ids': 0,
            'total_dist': 0.0, 'n_matches': 0,
            'n_gt': 0, 'n_pred': 0,
        }

    # Track ID mapping: gt_track_id -> last matched pred_track_id
    last_match = {}

    for fid in frame_ids:
        gts = gt_by_frame.get(fid, [])
        preds = pred_by_frame.get(fid, [])

        # Evaluate per class
        for cls_id in [0, 1, 2]:
            cls_gts = [g for g in gts if g['class_id'] == cls_id]
            cls_preds = [p for p in preds if p['class_id'] == cls_id]

            # Filter moving only
            if moving_only and moving_ids is not None:
                cls_gts = [g for g in cls_gts if g['track_id'] in moving_ids]

            # Get distance threshold for this class
            thre = dist_thre if dist_thre is not None else CLASS_DIST_THRE[cls_id]

            m = metrics[cls_id]
            m_all = metrics[None]

            m['n_gt'] += len(cls_gts)
            m['n_pred'] += len(cls_preds)
            m_all['n_gt'] += len(cls_gts)
            m_all['n_pred'] += len(cls_preds)

            if not cls_gts and not cls_preds:
                continue

            if not cls_gts:
                m['fp'] += len(cls_preds)
                m_all['fp'] += len(cls_preds)
                continue

            if not cls_preds:
                m['fn'] += len(cls_gts)
                m_all['fn'] += len(cls_gts)
                continue

            # Compute distance matrix
            n_gt, n_pred = len(cls_gts), len(cls_preds)
            dist_mat = np.zeros((n_gt, n_pred))
            for i, g in enumerate(cls_gts):
                for j, p in enumerate(cls_preds):
                    dist_mat[i, j] = np.sqrt((g['x'] - p['x'])**2 + (g['y'] - p['y'])**2)

            # Matching
            matched_gt = set()
            matched_pred = set()

            if matching == 'hungarian':
                # Hungarian (optimal) matching - same as motmetrics
                cost = dist_mat.copy()
                cost[cost >= thre] = 1e6  # infeasible
                row_ind, col_ind = linear_sum_assignment(cost)
                match_pairs = []
                for i, j in zip(row_ind, col_ind):
                    if dist_mat[i, j] < thre:
                        match_pairs.append((dist_mat[i, j], i, j))
            else:
                # Greedy matching (sort by distance, pick closest first)
                pairs = []
                for i in range(n_gt):
                    for j in range(n_pred):
                        if dist_mat[i, j] < thre:
                            pairs.append((dist_mat[i, j], i, j))
                pairs.sort()
                match_pairs = []
                for dist, i, j in pairs:
                    if i in matched_gt or j in matched_pred:
                        continue
                    match_pairs.append((dist, i, j))
                    matched_gt.add(i)
                    matched_pred.add(j)
                matched_gt.clear()
                matched_pred.clear()

            for dist, i, j in match_pairs:
                matched_gt.add(i)
                matched_pred.add(j)

                m['tp'] += 1
                m_all['tp'] += 1
                m['total_dist'] += dist
                m_all['total_dist'] += dist
                m['n_matches'] += 1
                m_all['n_matches'] += 1

                # Check ID switch
                gt_tid = cls_gts[i]['track_id']
                pred_tid = cls_preds[j]['track_id']
                key = (cls_id, gt_tid)
                if key in last_match and last_match[key] != pred_tid:
                    m['ids'] += 1
                    m_all['ids'] += 1
                last_match[key] = pred_tid

            # Unmatched
            fn = n_gt - len(matched_gt)
            fp = n_pred - len(matched_pred)
            m['fn'] += fn
            m['fp'] += fp
            m_all['fn'] += fn
            m_all['fp'] += fp

    return metrics


def print_metrics(metrics, label=""):
    """Print formatted metrics table."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  {'Class':<12} {'MOTA':>7} {'MOTP':>7} {'IDS':>5} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>6} {'Rec':>6}")
    print(f"  {'-'*64}")

    for cls_id in [None, 0, 1, 2]:
        m = metrics[cls_id]
        name = 'OVERALL' if cls_id is None else CLASS_NAMES[cls_id]

        if m['n_gt'] == 0:
            print(f"  {name:<12} {'N/A':>7}")
            continue

        mota = 1.0 - (m['fp'] + m['fn'] + m['ids']) / max(m['n_gt'], 1)
        motp = m['total_dist'] / max(m['n_matches'], 1)
        prec = m['tp'] / max(m['tp'] + m['fp'], 1)
        rec = m['tp'] / max(m['tp'] + m['fn'], 1)

        print(f"  {name:<12} {mota:>7.3f} {motp:>7.3f} {m['ids']:>5d} "
              f"{m['tp']:>6d} {m['fp']:>6d} {m['fn']:>6d} {prec:>6.3f} {rec:>6.3f}")

    print()


def main():
    split = 'val'
    gt_by_frame, frame_ids = load_gt(split)
    total_gt = sum(len(v) for v in gt_by_frame.values())
    moving_ids = find_moving_track_ids(gt_by_frame, frame_ids, move_thre=1.0)
    total_moving_gt = sum(
        len([o for o in gt_by_frame.get(fid, []) if o['track_id'] in moving_ids])
        for fid in frame_ids
    )
    print(f"GT: {len(frame_ids)} frames, {total_gt} objects (3 classes)")
    print(f"Moving tracks: {len(moving_ids)} IDs, {total_moving_gt} instances")

    csv_paths = sys.argv[1:] if len(sys.argv) > 1 else [
        './results/vod/val_det_baseline.csv',
        './results/vod/val_det_de_fastpoly.csv',
    ]

    for path in csv_paths:
        if not os.path.exists(path):
            print(f"  Skip: {path} not found")
            continue
        name = os.path.basename(path).replace('.csv', '')
        pred = load_predictions(path)

        # All objects, class-specific thresholds
        metrics_all = evaluate(gt_by_frame, pred, frame_ids)
        print_metrics(metrics_all, label=f"{name} [ALL objects, class-specific dist]")

        # Moving only
        metrics_mov = evaluate(gt_by_frame, pred, frame_ids,
                               moving_only=True, moving_ids=moving_ids)
        print_metrics(metrics_mov, label=f"{name} [MOVING only, class-specific dist]")

    # Also run with old 2m uniform threshold for reference
    print(f"\n{'#'*70}")
    print(f"  Reference: uniform 2m threshold (old protocol)")
    print(f"{'#'*70}")
    for path in csv_paths:
        if not os.path.exists(path):
            continue
        name = os.path.basename(path).replace('.csv', '')
        pred = load_predictions(path)
        metrics_old = evaluate(gt_by_frame, pred, frame_ids, dist_thre=2.0)
        print_metrics(metrics_old, label=f"{name} [uniform 2m]")


if __name__ == '__main__':
    main()
