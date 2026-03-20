"""
Pre-compute radar v_r for each CenterPoint detection on nuScenes val set.

For each sample:
1. Load radar points from all 5 sensors, transform to ego frame
2. For each detection box, find radar points inside (BEV matching)
3. Compute v_r from radar velocity: v_r = (x*vx_comp + y*vy_comp) / r

Output: JSON file mapping sample_token -> list of {vr, has_valid_vr} per detection.
"""
import json, os, sys, time
import numpy as np
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from pyquaternion import Quaternion

NUSC_PATH = os.environ.get('NUSCENES_ROOT', './data/nuscenes')
DETECTION_PATH = 'data/detector/val/val_centerpoint_new.json'
OUTPUT_PATH = 'data/nusc_radar_vr.json'

# All 5 radar sensors in nuScenes
RADAR_SENSORS = [
    'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT',
    'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT'
]


def load_radar_points_ego(nusc, sample, nsweeps=3):
    """Load radar points from all 5 sensors, transform to ego frame at sample timestamp.

    Returns: (N, 5) array with columns [x, y, z, vx_comp, vy_comp] in ego frame.
    """
    # Get ego pose at sample timestamp (use LIDAR_TOP as reference)
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    ego_trans = np.array(ego_pose['translation'])
    ego_rot = Quaternion(ego_pose['rotation'])

    all_points = []

    for sensor_name in RADAR_SENSORS:
        if sensor_name not in sample['data']:
            continue

        sd_token = sample['data'][sensor_name]
        sd = nusc.get('sample_data', sd_token)

        # Load current + past sweeps
        current_sd = sd
        for sweep_idx in range(nsweeps):
            if current_sd is None:
                break

            # Load radar point cloud (18 fields)
            pc_path = os.path.join(NUSC_PATH, current_sd['filename'])
            if not os.path.exists(pc_path):
                break

            pc = RadarPointCloud.from_file(pc_path)
            points = pc.points.T  # (N, 18)

            if len(points) == 0:
                if current_sd['prev'] != '':
                    current_sd = nusc.get('sample_data', current_sd['prev'])
                else:
                    break
                continue

            # Extract position, compensated velocity, and RCS
            # RadarPointCloud fields: x, y, z, dyn_prop, id, rcs, vx, vy, vx_comp, vy_comp, ...
            x = points[:, 0]
            y = points[:, 1]
            z = points[:, 2]
            rcs = points[:, 5]      # radar cross section (dBsm)
            vx_comp = points[:, 8]  # velocity compensated for ego motion
            vy_comp = points[:, 9]

            # Transform from sensor frame to ego frame
            # sensor -> ego
            cs = nusc.get('calibrated_sensor', current_sd['calibrated_sensor_token'])
            sensor_rot = Quaternion(cs['rotation'])
            sensor_trans = np.array(cs['translation'])

            # Rotate and translate positions
            pts_sensor = np.stack([x, y, z], axis=1)  # (N, 3)
            pts_ego_local = np.array([sensor_rot.rotate(p) + sensor_trans for p in pts_sensor])

            # Rotate velocities (no translation for velocity)
            vel_sensor = np.stack([vx_comp, vy_comp, np.zeros_like(vx_comp)], axis=1)
            vel_ego_local = np.array([sensor_rot.rotate(v) for v in vel_sensor])

            # If this is a sweep (not the key sample), transform from that sweep's ego to ref ego
            if sweep_idx > 0:
                sweep_ego_pose = nusc.get('ego_pose', current_sd['ego_pose_token'])
                sweep_trans = np.array(sweep_ego_pose['translation'])
                sweep_rot = Quaternion(sweep_ego_pose['rotation'])

                # sweep ego -> global -> ref ego
                pts_global = np.array([sweep_rot.rotate(p) + sweep_trans for p in pts_ego_local])
                pts_ref_ego = np.array([ego_rot.inverse.rotate(p - ego_trans) for p in pts_global])

                vel_global = np.array([sweep_rot.rotate(v) for v in vel_ego_local])
                vel_ref_ego = np.array([ego_rot.inverse.rotate(v) for v in vel_global])
            else:
                pts_ref_ego = pts_ego_local
                vel_ref_ego = vel_ego_local

            # Combine (add RCS as 6th column)
            combined = np.column_stack([
                pts_ref_ego[:, 0], pts_ref_ego[:, 1], pts_ref_ego[:, 2],
                vel_ref_ego[:, 0], vel_ref_ego[:, 1], rcs
            ])
            all_points.append(combined)

            # Move to prev sweep
            if current_sd['prev'] != '':
                current_sd = nusc.get('sample_data', current_sd['prev'])
            else:
                break

    if not all_points:
        return np.zeros((0, 6))

    return np.vstack(all_points)


def point_in_box_bev(px, py, box_x, box_y, box_w, box_l, box_yaw):
    """Check if point (px, py) is inside box in BEV (bird's eye view).

    Box defined by center (box_x, box_y), size (box_w, box_l), and yaw angle.
    """
    # Translate to box center
    dx = px - box_x
    dy = py - box_y

    # Rotate to box frame
    cos_yaw = np.cos(-box_yaw)
    sin_yaw = np.sin(-box_yaw)
    local_x = dx * cos_yaw - dy * sin_yaw
    local_y = dx * sin_yaw + dy * cos_yaw

    # Check if inside (w = width, l = length)
    # Add margin for radar noise
    margin = 1.0  # 1m margin
    half_w = box_w / 2 + margin
    half_l = box_l / 2 + margin

    return (np.abs(local_x) <= half_l) & (np.abs(local_y) <= half_w)


def det_to_ego(det, ego_pose):
    """Transform detection from global frame to ego frame."""
    ego_trans = np.array(ego_pose['translation'])
    ego_rot = Quaternion(ego_pose['rotation'])

    # Position: global -> ego
    pos_global = np.array(det['translation'])
    pos_ego = ego_rot.inverse.rotate(pos_global - ego_trans)

    # Velocity: global -> ego (rotation only)
    vel_global = np.array([det['velocity'][0], det['velocity'][1], 0.0])
    vel_ego = ego_rot.inverse.rotate(vel_global)

    # Yaw: global -> ego
    det_rot = Quaternion(det['rotation'])
    det_rot_ego = ego_rot.inverse * det_rot
    yaw_ego = det_rot_ego.yaw_pitch_roll[0]

    return {
        'x': pos_ego[0], 'y': pos_ego[1], 'z': pos_ego[2],
        'w': det['size'][0], 'l': det['size'][1], 'h': det['size'][2],
        'yaw': yaw_ego,
        'vx': vel_ego[0], 'vy': vel_ego[1],
    }


def main():
    print("Loading nuScenes database...")
    nusc = NuScenes(version='v1.0-trainval', dataroot=NUSC_PATH, verbose=True)

    print(f"Loading detections from {DETECTION_PATH}...")
    with open(DETECTION_PATH) as f:
        det_data = json.load(f)
    detections = det_data['results']
    print(f"  {len(detections)} frames")

    result = {}
    total_dets = 0
    total_with_vr = 0

    for sample_token in tqdm(detections, desc="Processing"):
        sample = nusc.get('sample', sample_token)
        dets = detections[sample_token]

        if not dets:
            result[sample_token] = []
            continue

        # Get ego pose for this sample
        lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])

        # Load radar points in ego frame
        radar_pts = load_radar_points_ego(nusc, sample, nsweeps=3)
        # radar_pts: (N, 5) = [x, y, z, vx, vy] in ego frame

        frame_results = []
        for det in dets:
            total_dets += 1

            # Transform detection to ego frame
            det_ego = det_to_ego(det, ego_pose)

            if len(radar_pts) == 0:
                frame_results.append({'vr': 0.0, 'has_valid_vr': False, 'rcs': 0.0, 'has_valid_rcs': False})
                continue

            # Find radar points inside this box (BEV)
            inside = point_in_box_bev(
                radar_pts[:, 0], radar_pts[:, 1],
                det_ego['x'], det_ego['y'],
                det_ego['w'], det_ego['l'], det_ego['yaw']
            )

            if np.sum(inside) == 0:
                frame_results.append({'vr': 0.0, 'has_valid_vr': False, 'rcs': 0.0, 'has_valid_rcs': False})
                continue

            # Compute v_r for each radar point inside the box
            pts_inside = radar_pts[inside]
            rx, ry = pts_inside[:, 0], pts_inside[:, 1]
            rvx, rvy = pts_inside[:, 3], pts_inside[:, 4]
            rcs_inside = pts_inside[:, 5]
            r = np.sqrt(rx**2 + ry**2)
            r = np.maximum(r, 1e-6)

            # v_r = (x*vx + y*vy) / r (radial velocity from ego to point)
            vr_points = (rx * rvx + ry * rvy) / r

            # Use median v_r (robust to outliers)
            vr_median = float(np.median(vr_points))

            # Use median RCS (robust to outliers)
            rcs_median = float(np.median(rcs_inside))

            frame_results.append({
                'vr': vr_median,
                'has_valid_vr': True,
                'n_pts': int(np.sum(inside)),
                'rcs': rcs_median,
                'has_valid_rcs': True,
            })
            total_with_vr += 1

        result[sample_token] = frame_results

    print(f"\nTotal detections: {total_dets}")
    print(f"With radar v_r: {total_with_vr} ({100*total_with_vr/max(total_dets,1):.1f}%)")

    print(f"Saving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(result, f)
    print("Done!")


if __name__ == '__main__':
    main()
