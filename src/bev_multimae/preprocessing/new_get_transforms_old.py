from collections import defaultdict, deque
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from scipy.spatial.transform import Rotation as R
import os

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

    frames = set()
    for p, c in transforms.keys():
        frames.add(p)
        frames.add(c)

    # print("Available frames:")
    # for f in sorted(frames):
    #     print(" ", f)

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

    raise KeyError("No path")

def get_all_tfs(cfg, right=True):

    transforms = {}
    folder = cfg.bags_path
    mcap = cfg.mcap_path
    path = mcap

    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])

        for _, channel, _, ros_msg in reader.iter_decoded_messages():
            if channel.topic != "/tf_static":
                continue

            for tf in ros_msg.transforms:
                # print(tf.header.frame_id, tf.child_frame_id)
                transforms[(tf.header.frame_id, tf.child_frame_id)] = transform_to_matrix(tf.transform)

    if right: direction = 'right' 
    else: direction = 'left'

    T_cam_ego = find_transform(
        transforms,
        "sensor_base_link",
        f"camera_front_{direction}_optical_frame",
        print_chain=False
    )
    T_rad_ego = find_transform(
        transforms,
        "sensor_base_link",
        f"radar_front_{direction}",
        print_chain=False
    )
    T_rad_cam = find_transform(
        transforms,
        f"camera_front_{direction}_optical_frame",
        f"radar_front_{direction}",
        print_chain=False
    )
    T_lid_cam = find_transform(
        transforms,
        f"camera_front_{direction}_optical_frame",
        "lidar_front_top/laser",
        print_chain=False
    )
    T_lid_ego = find_transform(
        transforms,
        "sensor_base_link",
        "lidar_front_top/laser",
        print_chain=False
    )

    return T_cam_ego, T_rad_ego, T_rad_cam, T_lid_cam, T_lid_ego