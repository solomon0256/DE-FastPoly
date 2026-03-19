"""
visualize_cases.py - Generate BEV qualitative figures for DE-FastPoly paper.

Success case: Frame 04659, Cyclist, GT track_id=555
  Baseline has ID switch (nearest pred changes 1->0); DE-FastPoly consistent.
  v_r = -4.30 m/s (strong Doppler signal).

Failure case: Frame 08362, Car, GT track_id=100
  DE-FastPoly introduces ID switch; Baseline is correct.
  |v_r| ~ 0.00 (lateral motion, no useful Doppler).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import csv

# ── Paths ──────────────────────────────────────────────────────────────────
LABEL_DIR = os.environ.get('VOD_LABEL_DIR', './data/label_2')
BASELINE_CSV = './results/vod/val_baseline.csv'
DE_CSV = './results/vod/val_de_fastpoly.csv'
OUT_DIR = './figures'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Camera-to-BEV transform ───────────────────────────────────────────────
_TR = np.array([
    [-0.013857, -0.9997468, 0.01772762, 0.05283124],
    [0.10934269, -0.01913807, -0.99381983, 0.98100483],
    [0.99390751, -0.01183297, 0.1095802, 1.44445002],
])
_R = _TR[:, :3]
_t = _TR[:, 3]
_R_inv = np.linalg.inv(_R)


def cam_to_bev(x_cam, y_cam, z_cam):
    p_cam = np.array([x_cam, y_cam, z_cam])
    p_radar = _R_inv @ (p_cam - _t)
    return p_radar[0], p_radar[1]


# ── Data loading ───────────────────────────────────────────────────────────
def load_tracking_csv(csv_path):
    """Return dict: frame_id -> list of (track_id, cx, cy, class_name)."""
    data = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = int(row["frame_id"])
            tid = int(row["track_id"])
            cx = float(row["cx"])
            cy = float(row["cy"])
            cls = row["class_name"]
            data.setdefault(fid, []).append((tid, cx, cy, cls))
    return data


def load_gt_object(gt_track_id, frame_ids, class_name):
    """Load GT positions (BEV) for a specific track across frames."""
    positions = {}
    for fid in frame_ids:
        label_path = os.path.join(LABEL_DIR, f"{fid:05d}.txt")
        if not os.path.exists(label_path):
            continue
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 16:
                    continue
                cls = parts[0]
                tid = int(float(parts[1]))
                if cls != class_name or tid != gt_track_id:
                    continue
                x_cam = float(parts[11])
                y_cam = float(parts[12])
                z_cam = float(parts[13])
                bx, by = cam_to_bev(x_cam, y_cam, z_cam)
                positions[fid] = (bx, by)
    return positions


def get_nearest_track_per_frame(tracking_data, frame_ids, gt_positions,
                                class_filter, max_dist=4.0, min_frames=2):
    """For each frame, find the closest predicted track to GT.
    Only return tracks that appear in at least min_frames frames."""
    tracks = {}
    for fid in frame_ids:
        if fid not in tracking_data or fid not in gt_positions:
            continue
        gx, gy = gt_positions[fid]
        best_tid, best_cx, best_cy, best_dist = None, None, None, 1e9
        for tid, cx, cy, cls in tracking_data[fid]:
            if cls != class_filter:
                continue
            dist = np.hypot(cx - gx, cy - gy)
            if dist < best_dist:
                best_dist = dist
                best_tid, best_cx, best_cy = tid, cx, cy
        if best_tid is not None and best_dist < max_dist:
            tracks.setdefault(best_tid, {})[fid] = (best_cx, best_cy)
    # Filter out tracks appearing in too few frames
    tracks = {tid: fd for tid, fd in tracks.items() if len(fd) >= min_frames}
    return tracks


# ── Color palette ──────────────────────────────────────────────────────────
TRACK_COLORS = {}
_BASE_PALETTE = [
    "#e74c3c",  # red
    "#3498db",  # blue
    "#f39c12",  # orange
    "#9b59b6",  # purple
    "#1abc9c",  # teal
    "#e67e22",  # dark orange
    "#2980b9",  # darker blue
    "#c0392b",  # dark red
    "#16a085",  # dark teal
    "#8e44ad",  # dark purple
    "#d35400",  # burnt orange
    "#2c3e50",  # dark grey
    "#f1c40f",  # yellow
]


def get_track_color(tid):
    if tid not in TRACK_COLORS:
        idx = len(TRACK_COLORS) % len(_BASE_PALETTE)
        TRACK_COLORS[tid] = _BASE_PALETTE[idx]
    return TRACK_COLORS[tid]


# ── Plotting ───────────────────────────────────────────────────────────────
def plot_case(ax, tracking_data, gt_positions, frame_ids, class_filter,
              title, gt_track_id, max_dist=4.0, annotate_vr=None,
              highlight_frame=None, id_switch_info=None,
              label_offset_gt=(6, 6), label_offset_pred=(6, -7),
              info_box_pos='top-right', skip_pred_labels=False):
    """Plot one BEV subplot."""
    tracks = get_nearest_track_per_frame(tracking_data, frame_ids,
                                         gt_positions, class_filter, max_dist)

    sorted_frames = sorted(frame_ids)
    gt_frames = [f for f in sorted_frames if f in gt_positions]

    # ── GT trajectory ──
    gt_xs = [gt_positions[f][0] for f in gt_frames]
    gt_ys = [gt_positions[f][1] for f in gt_frames]
    ax.plot(gt_xs, gt_ys, '--', color='#27ae60', linewidth=1.0, zorder=2)

    n_gt = len(gt_frames)
    for i, fid in enumerate(gt_frames):
        gx, gy = gt_positions[fid]
        alpha = 0.3 + 0.7 * (i / max(n_gt - 1, 1))
        ax.scatter(gx, gy, s=40, facecolors='none', edgecolors='#27ae60',
                   linewidths=1.3, zorder=4, marker='o', alpha=alpha)

    # ── Predicted tracks ──
    for tid, frames_dict in sorted(tracks.items()):
        color = get_track_color(tid)
        fids_sorted = sorted(frames_dict.keys())
        txs = [frames_dict[f][0] for f in fids_sorted]
        tys = [frames_dict[f][1] for f in fids_sorted]
        ax.plot(txs, tys, '-', color=color, linewidth=0.8, alpha=0.6,
                zorder=3)
        n_pred = len(fids_sorted)
        for j, fid in enumerate(fids_sorted):
            px, py = frames_dict[fid]
            alpha = 0.3 + 0.7 * (j / max(n_pred - 1, 1))
            ax.scatter(px, py, s=25, color=color, marker='s', zorder=5,
                       edgecolors='black', linewidths=0.3, alpha=alpha)

    # ── Labels at highlight frame ──
    labeled_tids = set()
    if highlight_frame and highlight_frame in gt_positions:
        gx, gy = gt_positions[highlight_frame]
        ax.annotate(f"GT {gt_track_id}", (gx, gy),
                    textcoords="offset points", xytext=label_offset_gt,
                    fontsize=6, color='#1e8449', fontweight='bold',
                    zorder=10)
        if not skip_pred_labels:
            for tid, frames_dict in tracks.items():
                if highlight_frame in frames_dict:
                    px, py = frames_dict[highlight_frame]
                    color = get_track_color(tid)
                    ax.annotate(f"ID {tid}", (px, py),
                                textcoords="offset points",
                                xytext=label_offset_pred,
                                fontsize=6, color=color, fontweight='bold',
                                zorder=10)
                    labeled_tids.add(tid)
            # Also label tracks that are NOT at highlight frame
            for tid, frames_dict in tracks.items():
                if tid in labeled_tids:
                    continue
                last_fid = max(frames_dict.keys())
                px, py = frames_dict[last_fid]
                color = get_track_color(tid)
                ax.annotate(f"ID {tid}", (px, py),
                            textcoords="offset points",
                            xytext=label_offset_pred,
                            fontsize=6, color=color, fontweight='bold',
                            zorder=10)

    # ── GT motion arrow ──
    if len(gt_xs) >= 2:
        dx = gt_xs[-1] - gt_xs[-2]
        dy = gt_ys[-1] - gt_ys[-2]
        ax.annotate("", xy=(gt_xs[-1] + dx * 0.6, gt_ys[-1] + dy * 0.6),
                    xytext=(gt_xs[-1], gt_ys[-1]),
                    arrowprops=dict(arrowstyle='->', color='#27ae60',
                                    lw=1.0), zorder=6)

    # ── ID switch annotation ──
    if id_switch_info:
        sw_frame = id_switch_info.get("frame")
        old_tid = id_switch_info.get("old_tid")
        new_tid = id_switch_info.get("new_tid")
        label_xy = id_switch_info.get("label_offset", (10, 10))
        if sw_frame and sw_frame in gt_positions:
            gx, gy = gt_positions[sw_frame]
            circle = plt.Circle((gx, gy), 0.7, fill=False, edgecolor='red',
                                linewidth=1.2, linestyle='--', zorder=8)
            ax.add_patch(circle)
            ax.annotate(f"IDS: {old_tid}\u2192{new_tid}",
                        (gx, gy),
                        textcoords="offset points", xytext=label_xy,
                        fontsize=5.5, color='red', fontweight='bold',
                        zorder=10,
                        arrowprops=dict(arrowstyle='->', color='red',
                                        lw=0.6, shrinkA=4, shrinkB=2))

    # ── Info box ──
    info_parts = [f"Frames {sorted_frames[0]}\u2013{sorted_frames[-1]}"]
    if annotate_vr is not None:
        info_parts.append(f"$v_r$ = {annotate_vr:.2f} m/s")
    if info_box_pos == 'top-left':
        ib_x, ib_y, ib_ha, ib_va = 0.03, 0.97, 'left', 'top'
    else:  # top-right
        ib_x, ib_y, ib_ha, ib_va = 0.97, 0.97, 'right', 'top'
    ax.text(ib_x, ib_y, "\n".join(info_parts), transform=ax.transAxes,
            fontsize=5.5, verticalalignment=ib_va, horizontalalignment=ib_ha,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#bbb', alpha=0.85),
            zorder=10)

    ax.set_title(title, fontsize=8, fontweight='bold', pad=4)
    ax.set_xlabel("x (m)", fontsize=6.5)
    ax.set_ylabel("y (m)", fontsize=6.5)
    ax.tick_params(labelsize=5.5)
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return tracks  # return for debugging


def sync_axes(ax1, ax2, margin=1.5):
    """Synchronize axis limits between two subplots with margin."""
    x1a, x1b = ax1.get_xlim()
    x2a, x2b = ax2.get_xlim()
    y1a, y1b = ax1.get_ylim()
    y2a, y2b = ax2.get_ylim()
    xmin = min(x1a, x2a) - margin
    xmax = max(x1b, x2b) + margin
    ymin = min(y1a, y2a) - margin
    ymax = max(y1b, y2b) + margin
    ax1.set_xlim(xmin, xmax)
    ax2.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)


def make_legend(fig):
    """Add a shared legend."""
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markeredgecolor='#27ae60',
               markerfacecolor='none', markersize=5, markeredgewidth=1.3,
               label='Ground Truth'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#888',
               markeredgecolor='black', markersize=4.5, markeredgewidth=0.3,
               label='Predicted'),
        Line2D([0], [0], linestyle='--', color='#27ae60', linewidth=1.0,
               label='GT Trajectory'),
        Line2D([0], [0], linestyle='-', color='#888', linewidth=0.8,
               alpha=0.6, label='Pred Trajectory'),
        Line2D([0], [0], marker='o', color='w', markeredgecolor='red',
               markerfacecolor='none', markersize=6, markeredgewidth=1.2,
               linestyle='--', label='ID Switch'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
               fontsize=5.5, frameon=True, edgecolor='#bbb',
               bbox_to_anchor=(0.5, -0.04),
               handletextpad=0.3, columnspacing=1.0)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("Loading tracking data...")
    baseline = load_tracking_csv(BASELINE_CSV)
    de_fp = load_tracking_csv(DE_CSV)

    # ══════════════════════════════════════════════════════════════════════
    # SUCCESS CASE: Cyclist, GT 555, frames ~4655-4662
    #
    # Baseline: track 5 follows GT 555 in frames 4656-4657 (dist ~0.7-1.3m),
    #   then track 0 takes over at frame 4658+ (dist ~1-2m) -> ID SWITCH 5->0
    # DE-FastPoly: track 0 is consistently used -> NO SWITCH
    # ══════════════════════════════════════════════════════════════════════
    print("Generating success case figure...")
    TRACK_COLORS.clear()

    success_frames = list(range(4655, 4663))
    gt_555 = load_gt_object(555, success_frames, "Cyclist")
    print(f"  GT 555: {len(gt_555)} frames loaded")

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.2))
    plt.subplots_adjust(wspace=0.45, bottom=0.18, top=0.82, left=0.13,
                        right=0.97)

    # Baseline: ID switch (5->0) at frame 4658
    t1 = plot_case(axes[0], baseline, gt_555, success_frames, "Cyclist",
                   "Baseline", gt_track_id=555, max_dist=5.0,
                   annotate_vr=-4.30, highlight_frame=4662,
                   id_switch_info={"frame": 4658, "old_tid": 5,
                                   "new_tid": 0, "label_offset": (8, -16)},
                   label_offset_gt=(5, 6), label_offset_pred=(5, 7),
                   info_box_pos='top-left', skip_pred_labels=True)
    print(f"  Baseline tracks found: {list(t1.keys())}")
    # Manually add clearer track ID labels with arrows
    axes[0].annotate("ID 0", xy=(23.5, 2.8), xytext=(21.5, 0.5),
                     fontsize=6, color=get_track_color(0), fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color=get_track_color(0),
                                     lw=0.6),
                     zorder=10)
    axes[0].annotate("ID 5", xy=(28.1, 3.1), xytext=(30.0, 5.0),
                     fontsize=6, color=get_track_color(5), fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color=get_track_color(5),
                                     lw=0.6),
                     zorder=10)

    # DE-FastPoly: consistent tracking (no IDS)
    t2 = plot_case(axes[1], de_fp, gt_555, success_frames, "Cyclist",
                   "DE-FastPoly", gt_track_id=555, max_dist=5.0,
                   annotate_vr=-4.30, highlight_frame=4659,
                   label_offset_gt=(5, -10), label_offset_pred=(5, -8),
                   info_box_pos='top-left')
    print(f"  DE-FastPoly tracks found: {list(t2.keys())}")
    # Add "No ID Switch" annotation on right panel
    axes[1].text(0.5, 0.05, "No ID Switch", transform=axes[1].transAxes,
                 fontsize=6, color='#27ae60', fontweight='bold',
                 ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#eafaf1',
                           edgecolor='#27ae60', alpha=0.9),
                 zorder=10)

    sync_axes(axes[0], axes[1], margin=2.0)

    fig.suptitle("(a) Success: Cyclist ($v_r$= \u22124.30 m/s)",
                 fontsize=7.5, fontweight='bold', y=0.98)
    make_legend(fig)

    out_path = os.path.join(OUT_DIR, "qualitative_success.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # ══════════════════════════════════════════════════════════════════════
    # FAILURE CASE: Cyclist, GT 1358, frames ~4903-4910
    #
    # Baseline: track changes (35 in early frames) but no IDS at frame 4907
    # DE-FastPoly: v_r drops from -5.6 to -2.82 at frame 4907 -> IDS 27->23
    # Limitation: momentary v_r noise causes Doppler association to misfire
    # ══════════════════════════════════════════════════════════════════════
    print("Generating failure case figure...")
    TRACK_COLORS.clear()

    failure_frames = list(range(4903, 4911))
    gt_1358 = load_gt_object(1358, failure_frames, "Cyclist")
    print(f"  GT 1358: {len(gt_1358)} frames loaded")

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.2))
    plt.subplots_adjust(wspace=0.45, bottom=0.18, top=0.82, left=0.13,
                        right=0.97)

    # Baseline: no IDS at frame 4907 (uses different tracks but stable in window)
    t3 = plot_case(axes[0], baseline, gt_1358, failure_frames, "Cyclist",
                   "Baseline", gt_track_id=1358, max_dist=4.0,
                   annotate_vr=-5.60, highlight_frame=4907,
                   label_offset_gt=(5, 6), label_offset_pred=(5, -8),
                   info_box_pos='top-right')
    print(f"  Baseline tracks found: {list(t3.keys())}")

    # DE-FastPoly: IDS 27->23 at frame 4907 due to v_r noise (-5.6 -> -2.82 m/s)
    t4 = plot_case(axes[1], de_fp, gt_1358, failure_frames, "Cyclist",
                   "DE-FastPoly", gt_track_id=1358, max_dist=4.0,
                   annotate_vr=-5.60, highlight_frame=4907,
                   id_switch_info={"frame": 4907, "old_tid": 27,
                                   "new_tid": 23, "label_offset": (10, 10)},
                   label_offset_gt=(5, 6), label_offset_pred=(5, -8),
                   info_box_pos='top-right')
    print(f"  DE-FastPoly tracks found: {list(t4.keys())}")

    sync_axes(axes[0], axes[1], margin=1.5)

    fig.suptitle("(b) Failure: $v_r$ noise ($-$5.6\u2192$-$2.82 m/s at fr.4907)",
                 fontsize=7.5, fontweight='bold', y=0.98)
    make_legend(fig)

    out_path = os.path.join(OUT_DIR, "qualitative_failure.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
