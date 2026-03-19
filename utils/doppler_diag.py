"""
Global diagnostics collector for Doppler experiments (Contribution A analysis).

This module is intentionally side-effect free for tracking behavior:
- When disabled, all record_* calls are no-ops.
- When enabled, it only accumulates statistics for later reporting.
"""

import json
import math
import os
from datetime import datetime
from typing import Dict, Optional

import numpy as np


def _empty_state() -> dict:
    return {
        'enabled': False,
        'dataset': None,
        'run_name': None,
        'output_path': None,
        'class_name_map': {},
        'meta': {},
        'per_class': {},
    }


_STATE = _empty_state()


def reset(config: Optional[dict] = None,
          run_name: Optional[str] = None,
          dataset: Optional[str] = None,
          class_name_map: Optional[Dict[int, str]] = None) -> None:
    """Reset global collector and configure whether diagnostics are enabled."""
    global _STATE
    _STATE = _empty_state()

    cfg = config or {}
    diag_cfg = cfg.get('doppler_diag', {})
    _STATE['enabled'] = bool(diag_cfg.get('enabled', False))
    _STATE['output_path'] = diag_cfg.get('output_path')
    _STATE['run_name'] = run_name
    _STATE['dataset'] = dataset
    _STATE['class_name_map'] = {int(k): str(v) for k, v in (class_name_map or {}).items()}
    _STATE['meta'] = {
        'created_at': datetime.utcnow().isoformat() + 'Z',
    }


def is_enabled() -> bool:
    return bool(_STATE.get('enabled', False))


def _cls_bucket(class_id: int) -> dict:
    key = str(int(class_id))
    if key not in _STATE['per_class']:
        _STATE['per_class'][key] = {
            'det_total': 0,
            'det_valid_vr': 0,
            'a_called': 0,
            'a_valid_vr': 0,
            'a_invalid_vr': 0,
            'a_enabled_but_ekf_path': 0,
            'vr_abs_values': [],
            'n_pts_values': [],
            'vr_meas_values': [],
            'vr_ref_values': [],
            'res_vr_values': [],
            'nis_vr_values': [],
            'nis_invalid_s_count': 0,
        }
    return _STATE['per_class'][key]


def record_det_obs(class_id: int,
                   has_valid_vr: bool,
                   radial_vel: Optional[float] = None,
                   n_pts: Optional[float] = None,
                   vr_ref: Optional[float] = None) -> None:
    """Record detection-side Doppler quality inputs."""
    if not is_enabled():
        return
    b = _cls_bucket(class_id)
    b['det_total'] += 1
    if has_valid_vr:
        b['det_valid_vr'] += 1

    if radial_vel is not None and math.isfinite(float(radial_vel)):
        b['vr_abs_values'].append(abs(float(radial_vel)))

    if n_pts is not None and math.isfinite(float(n_pts)):
        b['n_pts_values'].append(float(n_pts))

    # Correlation/RMSE should compare paired values; keep only finite pairs.
    if has_valid_vr and radial_vel is not None and vr_ref is not None:
        if math.isfinite(float(radial_vel)) and math.isfinite(float(vr_ref)):
            b['vr_meas_values'].append(float(radial_vel))
            b['vr_ref_values'].append(float(vr_ref))


def record_a_update(class_id: int, has_valid_vr: bool) -> None:
    """Record A-path entry in LinearKalmanFilter.update."""
    if not is_enabled():
        return
    b = _cls_bucket(class_id)
    b['a_called'] += 1
    if has_valid_vr:
        b['a_valid_vr'] += 1
    else:
        b['a_invalid_vr'] += 1


def record_a_ekf_path(class_id: int) -> None:
    """Record that A was enabled but EKF path was used."""
    if not is_enabled():
        return
    b = _cls_bucket(class_id)
    b['a_enabled_but_ekf_path'] += 1


def record_a_innovation(class_id: int, res_vr: float, s_vr: float) -> None:
    """Record v_r innovation and normalized innovation (NIS)."""
    if not is_enabled():
        return
    b = _cls_bucket(class_id)
    if res_vr is not None and math.isfinite(float(res_vr)):
        b['res_vr_values'].append(float(res_vr))
    if s_vr is None or (not math.isfinite(float(s_vr))) or float(s_vr) <= 0:
        b['nis_invalid_s_count'] += 1
        return
    nis = (float(res_vr) ** 2) / float(s_vr)
    if math.isfinite(nis):
        b['nis_vr_values'].append(float(nis))


def _series_stats(values: list) -> dict:
    if not values:
        return {'count': 0}
    arr = np.array(values, dtype=float)
    return {
        'count': int(arr.size),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'p50': float(np.percentile(arr, 50)),
        'p95': float(np.percentile(arr, 95)),
        'max': float(np.max(arr)),
    }


def _corr_rmse(meas: list, ref: list) -> dict:
    if len(meas) == 0 or len(ref) == 0 or len(meas) != len(ref):
        return {'pairs': 0}
    m = np.array(meas, dtype=float)
    r = np.array(ref, dtype=float)
    rmse = float(np.sqrt(np.mean((m - r) ** 2)))
    out = {'pairs': int(m.size), 'rmse': rmse}
    # Correlation is undefined for near-constant vectors.
    if m.size >= 2 and np.std(m) > 1e-12 and np.std(r) > 1e-12:
        out['corr'] = float(np.corrcoef(m, r)[0, 1])
    else:
        out['corr'] = None
    return out


def summarize(extra: Optional[dict] = None) -> dict:
    """Build serializable summary payload."""
    per_class = {}
    all_det_total = 0
    all_det_valid = 0
    all_a_called = 0
    all_a_valid = 0
    all_ekf_path = 0

    for cls_key, raw in sorted(_STATE['per_class'].items(), key=lambda x: int(x[0])):
        cls_id = int(cls_key)
        det_total = int(raw['det_total'])
        det_valid = int(raw['det_valid_vr'])
        a_called = int(raw['a_called'])
        a_valid = int(raw['a_valid_vr'])
        a_invalid = int(raw['a_invalid_vr'])
        a_ekf = int(raw['a_enabled_but_ekf_path'])

        all_det_total += det_total
        all_det_valid += det_valid
        all_a_called += a_called
        all_a_valid += a_valid
        all_ekf_path += a_ekf

        per_class[cls_key] = {
            'class_id': cls_id,
            'class_name': _STATE['class_name_map'].get(cls_id, str(cls_id)),
            'det_total': det_total,
            'det_valid_vr': det_valid,
            'valid_vr_ratio': (det_valid / det_total) if det_total > 0 else None,
            'a_called': a_called,
            'a_valid_vr': a_valid,
            'a_invalid_vr': a_invalid,
            'a_valid_ratio': (a_valid / a_called) if a_called > 0 else None,
            'a_enabled_but_ekf_path': a_ekf,
            'vr_abs_stats': _series_stats(raw['vr_abs_values']),
            'n_pts_stats': _series_stats(raw['n_pts_values']),
            'vr_meas_vs_ref': _corr_rmse(raw['vr_meas_values'], raw['vr_ref_values']),
            'res_vr_stats': _series_stats(raw['res_vr_values']),
            'nis_vr_stats': _series_stats(raw['nis_vr_values']),
            'nis_invalid_s_count': int(raw['nis_invalid_s_count']),
        }

    summary = {
        'enabled': bool(_STATE['enabled']),
        'dataset': _STATE['dataset'],
        'run_name': _STATE['run_name'],
        'meta': _STATE['meta'],
        'overall': {
            'det_total': all_det_total,
            'det_valid_vr': all_det_valid,
            'valid_vr_ratio': (all_det_valid / all_det_total) if all_det_total > 0 else None,
            'a_called': all_a_called,
            'a_valid_vr': all_a_valid,
            'a_valid_ratio': (all_a_valid / all_a_called) if all_a_called > 0 else None,
            'a_enabled_but_ekf_path': all_ekf_path,
        },
        'per_class': per_class,
        'extra': extra or {},
    }
    return summary


def dump(path: Optional[str] = None, extra: Optional[dict] = None) -> dict:
    """Write summary json if path is provided, and always return summary dict."""
    summary = summarize(extra=extra)
    out_path = path or _STATE.get('output_path')
    if out_path:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
    return summary

