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
    "/sensing/camera/front_left/compressed_image": "data/raw/camera/front_left",
    "/sensing/camera/front_left/camera_info":      "data/raw/camera/front_left",
    "/sensing/radar/front_left/raw/points":        "data/raw/radar/front_left",
    "/sensing/lidar/front_top/raw/points":          "data/raw/lidar/front_top",
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

def extract(cfg, input_path):
    saved_camera_info = set()
    bag_name = os.path.splitext(os.path.basename(input_path))[0]

    with open(input_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, ros_msg in reader.iter_decoded_messages(topics=list(TOPICS.keys())):
            sensor_dir = TOPICS[channel.topic]
            output_dir = os.path.join(bag_name, sensor_dir)
            output_dir = os.path.join(cfg.mcap_extract_path, output_dir)
            os.makedirs(output_dir, exist_ok=True)

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

            elif "lidar" in channel.topic and "points" in channel.topic:
                point_step = ros_msg.point_step
                data = np.frombuffer(ros_msg.data, dtype=np.uint8)
                n_points = len(data) // point_step
                raw = data.reshape(n_points, point_step)
                fields = {f.name: f.offset for f in ros_msg.fields}
                x = raw[:, fields["x"]:fields["x"]+4].view(np.float32).squeeze()
                y = raw[:, fields["y"]:fields["y"]+4].view(np.float32).squeeze()
                z = raw[:, fields["z"]:fields["z"]+4].view(np.float32).squeeze()
                xyz = np.stack([x, y, z], axis=1)
                output_path = os.path.join(output_dir, f"{timestamp}.bin")
                xyz.astype(np.float32).tofile(output_path)

            elif "points" in channel.topic:
                output_path = os.path.join(output_dir, f"{timestamp}.bin")
                with open(output_path, "wb") as pcd_file:
                    pcd_file.write(ros_msg.data)


@hydra.main(config_path="../../../configs", config_name="data", version_base=None)
def main(cfg: DictConfig) -> None:
    input_path = cfg.mcap_path
    if _mode == "list_topics":
        list_topics(input_path)
    elif _mode == "list_transforms":
        list_transforms(input_path)
    else:
        mcaps = [f for f in os.listdir(cfg.bags_path) if f.endswith('.mcap')]
        for i, mcap in enumerate(mcaps):
            extract(cfg, os.path.join(cfg.bags_path, mcap))
            print(f'finished with bag {i}')


if __name__ == "__main__":
    _mode = sys.argv.pop() if len(sys.argv) > 1 else "extract"
    main()