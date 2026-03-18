# mcap_reader.py

import sys
import os
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from scipy.spatial.transform import Rotation

import hydra
from omegaconf import DictConfig
from pathlib import Path
from hydra.utils import to_absolute_path

file_path = 'data/raw/agriRobotData.mcap'

TOPICS = {
    "/sensing/camera/front_right/compressed_image": "data/raw/camera/front_right",
    "/sensing/camera/front_right/camera_info":      "data/raw/camera/front_right",
    "/sensing/radar/front_right/raw/points":        "data/raw/radar/front_right",
}


def list_topics(input_path):
    with open(input_path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        if summary is None:
            print("No summary available.")
            return
        for channel_id, channel in summary.channels.items():
            schema = summary.schemas.get(channel.schema_id)
            print(f"  {channel.topic:60s}  [{schema.name if schema else 'unknown'}]")

def list_transforms(input_path, verbose=True):
    transforms = {}

    with open(input_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, ros_msg in reader.iter_decoded_messages(topics=["/tf_static"]):
            for transform in ros_msg.transforms:
                parent = transform.header.frame_id
                child = transform.child_frame_id

                if (parent, child) in transforms:
                    continue

                t = transform.transform.translation
                r = transform.transform.rotation

                rot = Rotation.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
                T = np.eye(4)
                T[:3, :3] = rot
                T[:3, 3] = [t.x, t.y, t.z]

                transforms[(parent, child)] = T

                if verbose:
                    euler = Rotation.from_quat([r.x, r.y, r.z, r.w]).as_euler('xyz', degrees=True)
                    print(f"  {parent} -> {child}")
                    print(f"    translation: x={t.x:.4f}  y={t.y:.4f}  z={t.z:.4f}")
                    print(f"    rotation (euler xyz deg): roll={euler[0]:.2f}  pitch={euler[1]:.2f}  yaw={euler[2]:.2f}")
                    print()

    return transforms

def get_camera_transform(mcap_path: str, transforms=None) -> np.ndarray:
    if transforms is None:
        transforms = list_transforms(mcap_path, verbose=False)

    chain = [
        ("base_link",                                        "chassis"),
        ("chassis",                                          "sensor_base_link"),
        ("sensor_base_link",                                 "bracket_front_right"),
        ("bracket_front_right",                              "bracket_camera_front_right"),
        ("bracket_camera_front_right",                       "bracket_camera_front_right_sensor_mounting_point"),
        ("bracket_camera_front_right_sensor_mounting_point", "nominal_camera_front_right"),
        ("nominal_camera_front_right",                       "camera_front_right"),
        ("camera_front_right",                               "camera_front_right_optical_frame"),
    ]

    return chain_transforms(transforms, chain)

def get_radar_transform(mcap_path: str, transforms=None) -> np.ndarray:
    if transforms is None:
        transforms = list_transforms(mcap_path, verbose=False)

    chain = [
        ("base_link",                 "chassis"),
        ("chassis",                   "sensor_base_link"),
        ("sensor_base_link",          "bracket_front_right"),
        ("bracket_front_right",       "nominal_radar_front_right"),
        ("nominal_radar_front_right", "radar_front_right"),
    ]

    return chain_transforms(transforms, chain)

def get_radar_to_camera_transform(mcap_path: str) -> np.ndarray:
    transforms      = list_transforms(mcap_path, verbose=False)
    T_base_to_radar = get_radar_transform(mcap_path, transforms)
    T_base_to_cam   = get_camera_transform(mcap_path, transforms)

    return np.linalg.inv(T_base_to_cam) @ T_base_to_radar

def get_transform(transforms, parent_frame, child_frame):
    key = (parent_frame, child_frame)
    if key not in transforms:
        available = "\n".join(f"  {p} -> {c}" for p, c in transforms.keys())
        raise KeyError(f"Transform {parent_frame} -> {child_frame} not found.\nAvailable:\n{available}")
    return transforms[key]

def chain_transforms(transforms, frame_chain):
    # Compose a sequence of transforms into a single matrix
    T = np.eye(4)
    for parent, child in frame_chain:
        T = T @ get_transform(transforms, parent, child)
    return T


def apply_transform(T, points_xyz):
    N = points_xyz.shape[0]
    pts_h = np.hstack([points_xyz, np.ones((N, 1))])
    return (T @ pts_h.T).T[:, :3]


def extract(input_path):
    saved_camera_info = set()

    with open(input_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, ros_msg in reader.iter_decoded_messages(topics=list(TOPICS.keys())):
            output_dir = TOPICS[channel.topic]
            timestamp = message.log_time

            if "compressed_image" in channel.topic:
                output_path = os.path.join(output_dir, f"{timestamp}.jpg")
                with open(output_path, "wb") as img_file:
                    img_file.write(ros_msg.data)

            elif "camera_info" in channel.topic:
                if channel.topic not in saved_camera_info:
                    output_path = os.path.join(output_dir, "camera_info.npz")
                    np.savez(
                        output_path,
                        K=np.array(ros_msg.k).reshape(3, 3),
                        D=np.array(ros_msg.d),
                        R=np.array(ros_msg.r).reshape(3, 3),
                        P=np.array(ros_msg.p).reshape(3, 4),
                        width=ros_msg.width,
                        height=ros_msg.height,
                        distortion_model=ros_msg.distortion_model,
                    )
                    saved_camera_info.add(channel.topic)
                    print(f"Saved camera info for {channel.topic}")

            elif "points" in channel.topic:
                output_path = os.path.join(output_dir, f"{timestamp}.bin")
                with open(output_path, "wb") as pcd_file:
                    pcd_file.write(ros_msg.data)


@hydra.main(config_path="../../../configs", config_name="data_config", version_base=None)
def main(cfg: DictConfig) -> None:
    input_path = cfg.mcap_path
    if _mode == "list_topics":
        list_topics(input_path)
    elif _mode == "list_transforms":
        list_transforms(input_path)
    else:
        extract(input_path)


if __name__ == "__main__":
    _mode = sys.argv.pop() if len(sys.argv) > 1 else "extract"
    main()