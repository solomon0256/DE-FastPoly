# DE-FastPoly

Official implementation of "DE-FastPoly: Doppler-Enhanced 3D Multi-Object Tracking with 4D Radar"

> [**DE-FastPoly: Doppler-Enhanced 3D Multi-Object Tracking with 4D Radar**](https://arxiv.org/abs/xxxx.xxxxx),
> Yuhe Wen, Linh Kästner, Jens Lambrecht
> *IEEE International Symposium on Industrial Electronics (ISIE), 2026*

## About

4D radar provides per-point radial velocity (Doppler), but most 3D trackers just throw it away. We built on top of [Fast-Poly](https://github.com/lixiaoyu2000/FastPoly) and added two simple modules that make use of it:

- **Doppler Velocity Init (DVI)** — uses the measured radial velocity to kick-start the Kalman filter at track birth, instead of assuming zero velocity.
- **Doppler-Aware Association (DA)** — mixes a Doppler similarity score into the association cost so that objects moving at different speeds don't get confused even when they overlap geometrically.

No extra latency, no architecture change needed. Just plug in.

## Architecture

```
  3D Detections + radar v_r
          |
          +---> DVI: project v_r to (vx, vy) at track birth
          |             |
          |        Kalman Filter (better init)
          |
          +---> DA: c = alpha * c_geo + beta * c_doppler
                        |
                   Hungarian matching
```

## Results

### VoD (oracle detection)

 Method       | MOTA  | IDS |
--------------|-------|-----|
 FastPoly     | 0.859 | 217 |
 **DE-FastPoly** | **0.893** | **165** |

MOTA +3.9%, IDS -24%

### VoD (PointPillars detector)

 Method       | MOTA  | IDS |
--------------|-------|-----|
 FastPoly     | 0.282 | 167 |
 **DE-FastPoly** | **0.299** | **154** |

MOTA +6.0% (relative), IDS -7.8%

## Getting Started

### 1. Environment

```
conda env create -f environment.yaml
conda activate fastpoly
```

### 2. Data

We test on [View of Delft (VoD)](https://intelligent-vehicles.org/datasets/view-of-delft/) and [nuScenes](https://www.nuscenes.org/).

For VoD, download the dataset and set:
```
export VOD_ROOT=/path/to/view_of_delft_PUBLIC
export VOD_LABEL_DIR=/path/to/label_2
```

For nuScenes, follow the data prep in [Fast-Poly](https://github.com/lixiaoyu2000/FastPoly#2-required-data) to get the detector output, token tables, and eval database. Then precompute radar velocities:
```
python precompute_nusc_radar_vr.py --dataroot /path/to/nuscenes --out_dir data/nusc_radar_vr
```

### 3. Run

```bash
# VoD with ground-truth boxes
python run_vod.py

# VoD with PointPillars detections
python run_vod_det.py

# nuScenes (same entry point as Fast-Poly)
python test.py
```

### 4. Eval

```bash
python eval_vod.py results/vod/val_de_fastpoly.csv
```

### 5. Scripts

Extra scripts under `scripts/` for ablation studies, visualization, etc.:

```bash
python scripts/reproduce_all.py      # reproduce all paper results (~88s)
python scripts/paper_ablation.py     # oracle ablation (Table II in paper)
python scripts/det_ablation_vod.py   # detector ablation (Table III)
python scripts/vr_statistics.py      # v_r distribution plot (Fig. 2)
python scripts/visualize_cases.py    # BEV qualitative cases (Fig. 3)
```

## What we changed (vs Fast-Poly)

| File | What |
|------|------|
| `tracking/nusc_tracker.py` | DA cost blending |
| `tracking/nusc_trajectory.py` | stores last measured v_r per track |
| `motion_module/motion_model.py` | DVI — velocity init from v_r |
| `motion_module/kalman_filter.py` | doppler-related KF config |
| `config/vod_config.yaml` | VoD config (new) |

## News

- 2026-03-15: Paper submitted to ISIE 2026.
- 2026-03-20: Code released.

## Contact

Feel free to reach out if you have questions or find bugs.

Yuhe Wen — wenyuhe03@gmail.com

## Citation

```
@inproceedings{wen2026defastpoly,
  title     = {DE-FastPoly: Doppler-Enhanced 3D Multi-Object Tracking with 4D Radar},
  author    = {Wen, Yuhe and K{\"a}stner, Linh and Lambrecht, Jens},
  booktitle = {IEEE International Symposium on Industrial Electronics (ISIE)},
  year      = {2026}
}
```

Also consider citing the original Fast-Poly, which this work builds on:
```
@article{li2024fast,
  title     = {Fast-Poly: A Fast Polyhedral Algorithm For 3D Multi-Object Tracking},
  author    = {Li, Xiaoyu and Liu, Dedong and Wu, Yitao and Wu, Xian and Zhao, Lijun and Gao, Jinghan},
  journal   = {IEEE Robotics and Automation Letters},
  year      = {2024},
  publisher = {IEEE}
}
```

## License

MIT
