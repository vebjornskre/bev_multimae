from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from scipy.spatial.transform import Rotation as R
import os

TF_STATIC_FILES = {
    "evt_0dzD3UAh6sdKnHTw.mcap": datetime(2025, 9, 10, 9, 37, 12, tzinfo=timezone.utc),
    "evt_0dyCz17eEVFSFoPD.mcap": datetime(2025, 9, 10, 11, 32, 1, tzinfo=timezone.utc),
    "evt_0dzD4cddqfhSyKxD.mcap": datetime(2025, 9, 10, 11, 44, 22, tzinfo=timezone.utc),
    "evt_0e8RkTcyPIIYG5ms.mcap": datetime(2025, 10, 2, 16, 5, 3, tzinfo=timezone.utc),
    "evt_0e3qZe9bbrfW2dr6.mcap": datetime(2025, 12, 2, 10, 42, 21, tzinfo=timezone.utc),
    "evt_0dyCeE6IuXW614wk.mcap": datetime(2025, 9, 10, 9, 45, 58, tzinfo=timezone.utc),
    "evt_0dyCcjgqVowqWW3y.mcap": datetime(2025, 9, 10, 9, 40, 17, tzinfo=timezone.utc),
    "evt_0dyCgPUWDSWVHohr.mcap": datetime(2025, 9, 10, 9, 52, 57, tzinfo=timezone.utc),
    "evt_0dzD5Pgi4RegF7gY.mcap": datetime(2025, 9, 10, 9, 50, 48, tzinfo=timezone.utc),
    "evt_0dyD87Zy4Ob5umNj.mcap": datetime(2025, 9, 10, 11, 33, 45, tzinfo=timezone.utc),
    "evt_0dzD5qjWKR6mAuMn.mcap": datetime(2025, 9, 10, 9, 48, 37, tzinfo=timezone.utc),
    "evt_0dyCjqcCwYrrnlQi.mcap": datetime(2025, 9, 10, 9, 54, 25, tzinfo=timezone.utc),
    "evt_0dzD2s7HuIGmDGkd.mcap": datetime(2025, 9, 10, 9, 30, 50, tzinfo=timezone.utc),
    "evt_0dzD6QByPLbhJBk2.mcap": datetime(2025, 9, 10, 9, 38, 53, tzinfo=timezone.utc),
    "evt_0dpi0fwKmPDYNXBa.mcap": datetime(2025, 7, 15, 15, 4, 19, tzinfo=timezone.utc),
    "evt_0dyJmvYazAMH4fsA.mcap": datetime(2025, 9, 10, 9, 32, 48, tzinfo=timezone.utc),
    "evt_0dzD3w3NCclnanOt.mcap": datetime(2025, 9, 10, 11, 48, 2, tzinfo=timezone.utc),
    "evt_0dyDEtaVQJ3Pepu9.mcap": datetime(2025, 9, 10, 11, 51, 9, tzinfo=timezone.utc),
    "evt_0dyD70lq51W3EjwD.mcap": datetime(2025, 9, 10, 11, 28, 7, tzinfo=timezone.utc),
    "evt_0e8RYiSLHB0LWQeG.mcap": datetime(2025, 10, 2, 17, 53, 2, tzinfo=timezone.utc),
    "evt_0dyKCDeSOrYmsHRY.mcap": datetime(2025, 9, 10, 11, 46, 9, tzinfo=timezone.utc),
    "evt_0dzD52ys8ZHByDj3.mcap": datetime(2025, 9, 10, 11, 26, 21, tzinfo=timezone.utc),
    "evt_0dyChJfiFRhHJ2AN.mcap": datetime(2025, 9, 10, 9, 57, 17, tzinfo=timezone.utc),
    "evt_0dyD9z6YPsRXnVyx.mcap": datetime(2025, 9, 10, 11, 35, 22, tzinfo=timezone.utc),
    "evt_0dyCtZKlVYIE42Y9.mcap": datetime(2025, 9, 10, 9, 58, 33, tzinfo=timezone.utc),
    "evt_0dyCxvkVIVIpkyLe.mcap": datetime(2025, 9, 10, 11, 30, 5, tzinfo=timezone.utc),
    "evt_0dyJyd30SYOwBKSH.mcap": datetime(2025, 9, 10, 11, 49, 53, tzinfo=timezone.utc),
}


def get_mcap_start_time(path: str) -> datetime:
    with open(path, "rb") as f:
        reader = make_reader(f)
        for _, _, message in reader.iter_messages():
            return datetime.fromtimestamp(message.log_time / 1e9, tz=timezone.utc)
    raise RuntimeError(f"No messages in {path}")


def find_tf_static_mcap(current_ts: datetime, bags_path: str) -> str:
    candidates = [
        (ts, fname) for fname, ts in TF_STATIC_FILES.items()
        if ts <= current_ts
    ]
    if not candidates:
        raise RuntimeError(f"No tf_static file found before {current_ts}")
    _, best_fname = max(candidates, key=lambda x: x[0])
    return os.path.join(bags_path, best_fname)


def transform_to_matrix(tf):
    t, q = tf.translation, tf.rotation
    quat = np.array([q.x, q.y, q.z, q.w])
    quat /= np.linalg.norm(quat)

    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat).as_matrix()
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def find_transform(transforms, source, target, print_chain=False):
    graph = defaultdict(list)
    for (p, c), T in transforms.items():
        graph[p].append((c, T))
        graph[c].append((p, np.linalg.inv(T)))

    queue = deque([(source, np.eye(4), [source])])
    visited = set()

    while queue:
        node, T_acc, path = queue.popleft()

        if node == target:
            if print_chain:
                print(" -> ".join(path))
            return T_acc

        if node in visited:
            continue
        visited.add(node)

        for nxt, T in graph[node]:
            queue.append((nxt, T_acc @ T, path + [nxt]))

    raise KeyError(f"No path from {source} to {target}")


def load_transforms(mcap_path: str) -> dict:
    transforms = {}
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, channel, _, ros_msg in reader.iter_decoded_messages(topics=["/tf_static"]):
            for tf in ros_msg.transforms:
                transforms[(tf.header.frame_id, tf.child_frame_id)] = transform_to_matrix(tf.transform)
    return transforms


def get_all_tfs(cfg, right=True):
    current_ts = get_mcap_start_time(cfg.mcap_path)
    tf_mcap = find_tf_static_mcap(current_ts, cfg.bags_path)
    transforms = load_transforms(tf_mcap)

    direction = 'right' if right else 'left'

    T_cam_ego = find_transform(transforms, "sensor_base_link", f"camera_front_{direction}_optical_frame")
    T_rad_ego = find_transform(transforms, "sensor_base_link", f"radar_front_{direction}")
    T_rad_cam = find_transform(transforms, f"camera_front_{direction}_optical_frame", f"radar_front_{direction}")
    T_lid_cam = find_transform(transforms, f"camera_front_{direction}_optical_frame", "lidar_front_top/laser")
    T_lid_ego = find_transform(transforms, "sensor_base_link", "lidar_front_top/laser")

    return T_cam_ego, T_rad_ego, T_rad_cam, T_lid_cam, T_lid_ego

def apply_transform(T, points_xyz):
    N = points_xyz.shape[0]
    pts_h = np.hstack([points_xyz, np.ones((N, 1))])
    return (T @ pts_h.T).T[:, :3]