"""
vr_statistics.py - VoD radial velocity (v_r) distribution analysis

Generates per-class statistics and publication-quality histograms of |v_r|
distribution in the VoD validation set. Used to justify threshold design
in DE-FastPoly (Doppler-Enhanced 3D MOT for 4D radar).

Usage:
    python vr_statistics.py
"""
import sys
import os
import numpy as np

# Reuse functions from run_vod.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_vod import (
    parse_kitti_label, read_radar_pointcloud, get_doppler_for_box,
    load_frame_ids, LABEL_DIR, VOD_CLASS_MAP,
)

# ===== Config =====
SPLIT = 'val'
CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist']
CLASS_IDS = {name: VOD_CLASS_MAP[name] for name in CLASS_NAMES}
THRESHOLDS = {'Car': 1.0, 'Pedestrian': 0.3, 'Cyclist': 0.5}
FIG_DIR = './figures'
FIG_PATH = os.path.join(FIG_DIR, 'vr_distribution.png')


def collect_vr_data():
    """Load all val GT boxes, extract radar v_r for each, return per-class arrays."""
    sequences = load_frame_ids(SPLIT)
    all_fids = [fid for seq in sequences for fid in seq]
    print(f"VoD {SPLIT}: {len(sequences)} sequences, {len(all_fids)} frames")

    # Per-class: list of |v_r| values (only for boxes with radar match)
    vr_by_class = {name: [] for name in CLASS_NAMES}
    # Per-class: total box count and matched count
    total_by_class = {name: 0 for name in CLASS_NAMES}
    matched_by_class = {name: 0 for name in CLASS_NAMES}

    for i, fid in enumerate(all_fids):
        if i % 200 == 0:
            print(f"  Processing frame {fid:05d} ({i+1}/{len(all_fids)})...")

        # Read GT labels
        label_path = os.path.join(LABEL_DIR, f'{fid:05d}.txt')
        dets = []
        if os.path.exists(label_path):
            with open(label_path) as lf:
                for line in lf:
                    det = parse_kitti_label(line)
                    if det is not None:
                        dets.append(det)

        if not dets:
            continue

        # Read radar point cloud
        pc = read_radar_pointcloud(fid)

        for det in dets:
            cls_name = det['class_name']
            total_by_class[cls_name] += 1

            vr, found, _ = get_doppler_for_box(
                pc, det['x'], det['y'], det['z'],
                det['w'], det['l'], det['h'], det['ry']
            )
            if found:
                matched_by_class[cls_name] += 1
                vr_by_class[cls_name].append(abs(vr))

    # Convert to numpy arrays
    for name in CLASS_NAMES:
        vr_by_class[name] = np.array(vr_by_class[name])

    return vr_by_class, total_by_class, matched_by_class


def compute_statistics(vr_by_class, total_by_class, matched_by_class):
    """Compute per-class statistics and print table."""
    stats = {}
    for name in CLASS_NAMES:
        vr = vr_by_class[name]
        n_total = total_by_class[name]
        n_matched = matched_by_class[name]
        thr = THRESHOLDS[name]

        if len(vr) == 0:
            stats[name] = None
            continue

        s = {
            'n_total': n_total,
            'n_matched': n_matched,
            'match_pct': 100.0 * n_matched / max(n_total, 1),
            'median': np.median(vr),
            'mean': np.mean(vr),
            'std': np.std(vr),
            'q25': np.percentile(vr, 25),
            'q75': np.percentile(vr, 75),
            'iqr': np.percentile(vr, 75) - np.percentile(vr, 25),
            'p90': np.percentile(vr, 90),
            'p95': np.percentile(vr, 95),
            'p99': np.percentile(vr, 99),
            'max': np.max(vr),
            'pct_gt_0.1': 100.0 * np.mean(vr > 0.1),
            'pct_gt_0.3': 100.0 * np.mean(vr > 0.3),
            'pct_gt_0.5': 100.0 * np.mean(vr > 0.5),
            'pct_gt_1.0': 100.0 * np.mean(vr > 1.0),
            'pct_gt_thr': 100.0 * np.mean(vr > thr),
            'threshold': thr,
        }
        stats[name] = s

    # Print table
    print("\n" + "=" * 90)
    print("VoD v_r Distribution Statistics (val split)")
    print("=" * 90)

    header = f"{'Metric':<28} {'Car':>18} {'Pedestrian':>18} {'Cyclist':>18}"
    print(header)
    print("-" * 90)

    rows = [
        ('GT boxes (total)',        'n_total',    'd'),
        ('Boxes w/ radar match',    'n_matched',  'd'),
        ('Match rate (%)',          'match_pct',  '.1f'),
        ('',                        None,         ''),
        ('Median |v_r| (m/s)',      'median',     '.4f'),
        ('Mean |v_r| (m/s)',        'mean',       '.4f'),
        ('Std |v_r| (m/s)',         'std',        '.4f'),
        ('Q25 (m/s)',               'q25',        '.4f'),
        ('Q75 (m/s)',               'q75',        '.4f'),
        ('IQR (m/s)',               'iqr',        '.4f'),
        ('90th pctile (m/s)',       'p90',        '.4f'),
        ('95th pctile (m/s)',       'p95',        '.4f'),
        ('99th pctile (m/s)',       'p99',        '.4f'),
        ('Max |v_r| (m/s)',         'max',        '.4f'),
        ('',                        None,         ''),
        ('% |v_r| > 0.1',          'pct_gt_0.1', '.1f'),
        ('% |v_r| > 0.3',          'pct_gt_0.3', '.1f'),
        ('% |v_r| > 0.5',          'pct_gt_0.5', '.1f'),
        ('% |v_r| > 1.0',          'pct_gt_1.0', '.1f'),
        ('% > class threshold',    'pct_gt_thr', '.1f'),
        ('Threshold (m/s)',         'threshold',  '.1f'),
    ]

    for label, key, fmt in rows:
        if key is None:
            print()
            continue
        vals = []
        for name in CLASS_NAMES:
            s = stats[name]
            if s is None:
                vals.append('N/A')
            elif fmt == 'd':
                vals.append(f"{int(s[key]):>18d}")
            else:
                vals.append(f"{s[key]:>18{fmt}}")
        print(f"{label:<28} {''.join(vals)}")

    print("=" * 90)
    return stats


def plot_violin(vr_by_class, stats):
    """Generate publication-quality violin plot (no threshold lines)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 9,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 8,
    })

    colors = {'Car': '#2166ac', 'Pedestrian': '#b2182b', 'Cyclist': '#1b7837'}

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    # Prepare data: use signed v_r (not |v_r|) for more information
    # But we collected |v_r|, so show |v_r| distribution clipped
    data = []
    labels = []
    cols = []
    for name in CLASS_NAMES:
        vr = vr_by_class[name]
        if len(vr) == 0:
            continue
        data.append(np.clip(vr, 0, 10.0))
        labels.append(name)
        cols.append(colors[name])

    parts = ax.violinplot(data, positions=range(len(data)), showmedians=True,
                          showextrema=False, widths=0.7)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(cols[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.8)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.5)

    # Add median + IQR text
    for i, name in enumerate(labels):
        s = stats[name]
        ax.text(i, s['p95'] + 0.3,
                f"med={s['median']:.2f}\nIQR={s['iqr']:.2f}",
                ha='center', va='bottom', fontsize=7, color='#333333')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel('$|v_r|$ (m/s)')
    ax.set_ylim(0, None)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title('Radial Velocity Distribution (VoD val)', fontsize=10)

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: {FIG_PATH}")
    plt.close(fig)


def main():
    print("Collecting v_r data from VoD val split...")
    vr_by_class, total_by_class, matched_by_class = collect_vr_data()

    stats = compute_statistics(vr_by_class, total_by_class, matched_by_class)

    print("\nGenerating violin plot...")
    plot_violin(vr_by_class, stats)

    # Summary for paper
    print("\n" + "=" * 70)
    print("KEY FINDINGS FOR PAPER:")
    print("=" * 70)
    for name in CLASS_NAMES:
        s = stats[name]
        if s is None:
            continue
        print(f"\n{name}:")
        print(f"  - {s['match_pct']:.1f}% of GT boxes have radar points")
        print(f"  - Median |v_r| = {s['median']:.4f} m/s  (low due to lateral motion)")
        print(f"  - 90th percentile = {s['p90']:.4f} m/s")
        print(f"  - Only {s['pct_gt_thr']:.1f}% exceed threshold ({s['threshold']} m/s)")
        print(f"  - Justification: most objects move laterally to radar -> small v_r")


if __name__ == '__main__':
    main()
