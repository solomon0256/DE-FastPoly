# DE-FastPoly

Doppler-Enhanced 3D Multi-Object Tracking with 4D Radar, built on [Fast-Poly](https://github.com/lixiaoyu2000/FastPoly).

> Yuhe Wen, Linh Kästner, Jens Lambrecht
> *Submitted to IEEE International Symposium on Industrial Electronics (ISIE), 2026*

## About

4D radar outputs per-point radial velocity (Doppler), but most 3D trackers don't use it. This repo adds four modules to Fast-Poly that exploit Doppler and other radar cues:

- **DVI** (Doppler Velocity Init): use measured v_r to initialize KF velocity at track birth instead of zero
- **DA** (Doppler-Aware Association): mix Doppler similarity into association cost, `c = alpha * c_geo + beta * c_doppler`
- **SAN** (Score-only Adaptive Noise): scale measurement noise R by detection confidence
- **RCSA** (RCS-Aware Association): radar cross section consistency as soft cue (pedestrian only)

~8ms/frame, no overhead vs vanilla Fast-Poly.

## Architecture

```
tracking/nusc_tracker.py       DA + RCSA association logic
tracking/nusc_trajectory.py    per-track v_r / RCS bookkeeping
motion_module/motion_model.py  DVI velocity init
motion_module/kalman_filter.py SAN adaptive noise, dynamic H
config/vod_config.yaml         VoD config (all modules enabled)
config/nusc_config.yaml        nuScenes config
```

## Results

### VoD — oracle detection

| Method | MOTA | MOTP | IDS | FP | FN |
|--------|------|------|-----|----|----|
| FastPoly (baseline) | 0.859 | 0.479 | 217 | 404 | 916 |
| **DE-FastPoly** | **0.893** | **0.464** | **165** | **411** | **537** |

### VoD — PointPillars detector

| Method | MOTA | IDS | FP | FN |
|--------|------|-----|----|----|
| FastPoly (baseline) | 0.282 | 167 | 1095 | 5555 |
| **DE-FastPoly** | **0.299** | **154** | 1118 | **5141** |

### nuScenes val

| Method | AMOTA | AMOTP | IDS |
|--------|-------|-------|-----|
| FastPoly (CenterPoint) | 0.7367 | 0.5765 | 414 |
| **DE-FastPoly** | **0.7367** | **0.5727** | **379** |

IDS drops by 35 (-8.5%) with no AMOTA regression.

## Getting Started

### 1. Environment

```bash
conda env create -f environment.yaml
conda activate fastpoly
```

or `pip install -r requirements.txt` if you don't use conda.

### 2. Data

We evaluate on [View-of-Delft (VoD)](https://intelligent-vehicles.org/datasets/view-of-delft/) and [nuScenes](https://www.nuscenes.org/).

**VoD**: download the dataset, then set environment variables:
```bash
export VOD_ROOT=/path/to/view_of_delft_PUBLIC
export VOD_LABEL_DIR=/path/to/label_2
```

**nuScenes**: follow [Fast-Poly's data prep](https://github.com/lixiaoyu2000/FastPoly#2-required-data) for detector outputs and eval database, then precompute radar velocities:
```bash
export NUSCENES_ROOT=/path/to/nuscenes
python precompute_nusc_radar_vr.py
```

### 3. Run tracking

```bash
# VoD with ground-truth boxes
python run_vod.py

# VoD with PointPillars detections
python run_vod_det.py

# nuScenes (same interface as Fast-Poly)
python test.py --nusc_path /path/to/nuscenes
```

### 4. Evaluate

```bash
python eval_vod.py results/vod/output.csv
```

### 5. Reproduce paper results

```bash
python scripts/reproduce_all.py      # all configs, ~88s
python scripts/paper_ablation.py     # oracle ablation (Table II)
python scripts/det_ablation_vod.py   # detector ablation (Table III)
python scripts/vr_statistics.py      # v_r distribution (Fig. 2)
python scripts/visualize_cases.py    # BEV visualization (Fig. 3)
```

## News

- 2026-03-20 Code released.
- 2026-03-15 Paper submitted to ISIE 2026.

## Contact

Yuhe Wen — wenyuhe03@gmail.com

## Acknowledgements

Based on [Fast-Poly](https://github.com/lixiaoyu2000/FastPoly) (Li et al., RAL 2024).
