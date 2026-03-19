# DE-FastPoly: Doppler-Enhanced Fast Polyhedral 3D Multi-Object Tracking

This is the official code for the paper:

> **DE-FastPoly: Doppler-Enhanced 3D Multi-Object Tracking with 4D Radar**
> Yuhe Wen, Linh Kästner, Jens Lambrecht
> *IEEE International Symposium on Industrial Electronics (ISIE), 2026*

DE-FastPoly extends [Fast-Poly](https://github.com/lixiaoyu2000/FastPoly) (RAL 2024) with two Doppler-based enhancements for 4D radar tracking:

- **Doppler Velocity Initialization (DVI)**: Projects measured radial velocity $v_r$ into Cartesian space to initialize Kalman filter velocity at track birth, replacing the zero-velocity assumption.
- **Doppler-Aware Association (DA)**: Blends a Doppler similarity term into the geometric association cost, improving data association for objects with similar geometry but different velocities.

Both modules are **plug-and-play** with zero runtime overhead (8.1 ms/frame).

## Architecture

```
Detection (3D boxes + radar v_r)
        │
        ├── DVI: v_r → (vx, vy) projection at track birth
        │         ↓
        │    Kalman Filter with better velocity init
        │
        └── DA: c_final = α · c_geo + β · c_doppler
                  ↓
             Hungarian / Greedy assignment
```

## Main Results

### View-of-Delft (VoD) — Oracle Detection

| Config | MOTA | MOTP | IDS | FP | FN |
|--------|------|------|-----|----|----|
| Baseline (FastPoly) | 0.859 | 0.718 | 217 | 404 | 548 |
| + DVI | 0.870 | 0.716 | 189 | 411 | 466 |
| + DA | 0.871 | 0.717 | 175 | 402 | 502 |
| **+ DVI + DA** | **0.893** | **0.717** | **165** | **385** | **454** |

**MOTA +3.9%, IDS -24.0%** vs. baseline.

### View-of-Delft (VoD) — PointPillars Detector

| Config | MOTA | MOTP | IDS | FP | FN |
|--------|------|------|-----|----|----|
| Baseline | 0.282 | 0.443 | 167 | 1095 | 2953 |
| **+ DVI + DA** | **0.299** | **0.449** | **154** | **1118** | **2760** |

**MOTA +6.0% relative, IDS -7.8%** vs. baseline.

### nuScenes — CenterPoint Detector

| Method | AMOTA | AMOTP | IDS |
|--------|-------|-------|-----|
| FastPoly (baseline) | 0.7367 | 0.5765 | 414 |
| **DE-FastPoly (DA only)** | **0.7367** | **0.5742** | **379** |

**IDS -8.5%** with AMOTA maintained. DVI is disabled on nuScenes because CenterPoint already provides velocity estimates.

## Modified Files (vs. FastPoly)

| File | Change |
|------|--------|
| `tracking/nusc_tracker.py` | Doppler-aware association cost blending (DA) |
| `tracking/nusc_trajectory.py` | `last_vr` state storage and update |
| `motion_module/motion_model.py` | Velocity initialization from $v_r$ (DVI) |
| `motion_module/kalman_filter.py` | Doppler config, dynamic H matrix, range-adaptive R |
| `config/vod_config.yaml` | VoD configuration with Doppler parameters |
| `config/nusc_config.yaml` | nuScenes configuration |
| `utils/doppler_diag.py` | Doppler diagnostics collector (optional) |

## Key Parameters

```yaml
# in config/vod_config.yaml → doppler section
assoc_alpha: 0.95        # weight for geometric cost
assoc_beta: 0.05         # weight for Doppler cost
doppler_sigma: 10.0      # Gaussian normalization sigma
use_doppler_init: True   # DVI: enable velocity init from v_r
use_doppler_assoc: True  # DA: enable Doppler association
vr_class_thre:           # min |v_r| to trust (per class)
  0: 1.0   # Car
  1: 0.3   # Pedestrian
  2: 0.5   # Cyclist
```

## Getting Started

### 1. Environment Setup

```bash
conda env create -f environment.yaml
conda activate fastpoly
```

### 2. Data Preparation

#### VoD Dataset
Download the [View of Delft dataset](https://intelligent-vehicles.org/datasets/view-of-delft/) and set the environment variable:
```bash
export VOD_ROOT=/path/to/view_of_delft_PUBLIC
export VOD_LABEL_DIR=/path/to/label_2
```

#### nuScenes Dataset
Follow the [Fast-Poly instructions](https://github.com/lixiaoyu2000/FastPoly#2-required-data) to prepare detector files, token tables, and evaluation database. Then precompute radar $v_r$:
```bash
python precompute_nusc_radar_vr.py --dataroot /path/to/nuscenes --out_dir data/nusc_radar_vr
```

### 3. Running

#### VoD — Oracle (GT) Tracking
```bash
python run_vod.py
```

#### VoD — Detector (PointPillars) Tracking
```bash
python run_vod_det.py
```

#### nuScenes Tracking
```bash
python test.py
```

### 4. Evaluation

#### VoD Evaluation
```bash
python eval_vod.py results/vod/val_de_fastpoly.csv
```

#### Ablation Studies
```bash
python paper_ablation.py       # Oracle ablation (Table II)
python det_ablation_vod.py     # Detector ablation (Table III)
```

#### Reproduce All Results
```bash
python reproduce_all.py        # ~88s, validates all configurations
```

### 5. Visualization
```bash
python vr_statistics.py        # v_r distribution analysis (Figure 2)
python visualize_cases.py      # BEV qualitative visualization (Figure 3)
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{wen2026defastpoly,
  title={DE-FastPoly: Doppler-Enhanced 3D Multi-Object Tracking with 4D Radar},
  author={Wen, Yuhe and K{\"a}stner, Linh and Lambrecht, Jens},
  booktitle={IEEE International Symposium on Industrial Electronics (ISIE)},
  year={2026}
}
```

Also cite the original Fast-Poly:
```bibtex
@article{li2024fast,
  title={Fast-Poly: A Fast Polyhedral Algorithm For 3D Multi-Object Tracking},
  author={Li, Xiaoyu and Liu, Dedong and Wu, Yitao and Wu, Xian and Zhao, Lijun and Gao, Jinghan},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  publisher={IEEE}
}
```

## Acknowledgements

This project is built upon [Fast-Poly](https://github.com/lixiaoyu2000/FastPoly) and [Poly-MOT](https://github.com/lixiaoyu2000/Poly-MOT). We thank the authors for their excellent work.

## License

This project is released under the [MIT License](LICENSE).
