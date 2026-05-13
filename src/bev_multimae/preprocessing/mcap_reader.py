# mcap_reader.py
import cv2
import sys
import os
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from scipy.spatial.transform import Rotation

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from hydra.utils import to_absolute_path
from multiprocessing import Pool

file_path = 'data/raw/agriRobotData.mcap'

TOPICS = {
    "/sensing/camera/front_right/compressed_image":               "camera/front_right",
    "/sensing/camera/front_right/camera_info":                    "camera/front_right",
    "/sensing/radar/front_right/raw/points":                      "radar/front_right",
    "/sensing/camera/front_left/compressed_image":                "camera/front_left",
    "/sensing/camera/front_left/camera_info":                     "camera/front_left",
    "/sensing/radar/front_left/raw/points":                       "radar/front_left",
    "/sensing/lidar/front_top/raw/points":                        "lidar/front_top",
    "/sensing/camera/front_right/compressed_image/gt_seg/human":  "seg/front_right",
    "/sensing/camera/front_left/compressed_image/gt_seg/human":   "seg/front_left",
    "/sensing/camera/front_right/compressed_image/gt_bbox/human": "bbox/front_right",
    "/sensing/camera/front_left/compressed_image/gt_bbox/human":  "bbox/front_left"
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

def save_seg_mask(ros_msg, output_dir, timestamp):
    H, W = 720, 1280
    mask = np.zeros((H, W), dtype=np.uint8)
    for ann in ros_msg.points:
        for p in ann.points:
            x, y = int(p.x), int(p.y)
            if 0 <= x < W and 0 <= y < H:
                mask[y, x] = 1
    np.save(os.path.join(output_dir, f"{timestamp}.npy"), mask)


def save_image(ros_msg, output_dir, timestamp):
    with open(os.path.join(output_dir, f"{timestamp}.jpg"), "wb") as f:
        f.write(ros_msg.data)


def save_camera_info(ros_msg, output_dir):
    np.savez(
        os.path.join(output_dir, "camera_info.npz"),
        K=np.array(ros_msg.k).reshape(3, 3),
        D=np.array(ros_msg.d),
        R=np.array(ros_msg.r).reshape(3, 3),
        P=np.array(ros_msg.p).reshape(3, 4),
        width=ros_msg.width,
        height=ros_msg.height,
        distortion_model=ros_msg.distortion_model,
    )


def save_lidar_points(ros_msg, output_dir, timestamp):
    step = ros_msg.point_step
    data = np.frombuffer(ros_msg.data, dtype=np.uint8).reshape(-1, step)
    fields = {f.name: f.offset for f in ros_msg.fields}
    xyz = np.stack([
        data[:, fields[ax]:fields[ax]+4].view(np.float32).squeeze()
        for ax in ("x", "y", "z")
    ], axis=1)
    xyz.astype(np.float32).tofile(os.path.join(output_dir, f"{timestamp}.bin"))


def save_radar_points(ros_msg, output_dir, timestamp):
    with open(os.path.join(output_dir, f"{timestamp}.bin"), "wb") as f:
        f.write(ros_msg.data)


def save_bboxes(ros_msg, output_dir, timestamp):
    boxes = []
    for ann in ros_msg.points:
        xs = [p.x for p in ann.points]
        ys = [p.y for p in ann.points]
        boxes.append([min(xs), min(ys), max(xs), max(ys)])
    np.save(os.path.join(output_dir, f"{timestamp}.npy"), np.array(boxes, dtype=np.float32))


def extract(cfg, input_path):
    saved_camera_info = set()
    created_dirs = set()
    bag_name = os.path.splitext(os.path.basename(input_path))[0]
    timestamp_log = {}  # topic -> which timestamp was used

    with open(input_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, ros_msg in reader.iter_decoded_messages(topics=list(TOPICS.keys())):
            topic = channel.topic
            output_dir = os.path.join(cfg["mcap_extract_path"], bag_name, TOPICS[topic])
            if output_dir not in created_dirs:
                os.makedirs(output_dir, exist_ok=True)
                created_dirs.add(output_dir)

            if hasattr(ros_msg, "header"):
                sensor_ts = (
                    ros_msg.header.stamp.sec * 1_000_000_000
                    + ros_msg.header.stamp.nanosec
                )
                diff_ms = abs(message.log_time - sensor_ts) / 1_000_000
                if diff_ms < 5000:
                    ts = sensor_ts
                    timestamp_log[topic] = "sensor_timestamp"
                else:
                    ts = message.log_time
                    timestamp_log[topic] = "log_time (sensor stamp was invalid)"
            else:
                ts = message.log_time
                timestamp_log[topic] = "log_time (no header)"

            if "gt_bbox" in topic:
                save_bboxes(ros_msg, output_dir, ts)
            elif "gt_seg" in topic:
                save_seg_mask(ros_msg, output_dir, ts)
            elif "compressed_image" in topic:
                save_image(ros_msg, output_dir, ts) 
            elif "camera_info" in topic:
                if topic not in saved_camera_info:
                    save_camera_info(ros_msg, output_dir)
                    saved_camera_info.add(topic)
            elif "lidar" in topic:
                save_lidar_points(ros_msg, output_dir, ts)
            elif "points" in topic:
                save_radar_points(ros_msg, output_dir, ts)

    # write timestamp info to txt file in bag root folder
    bag_root = os.path.join(cfg["mcap_extract_path"], bag_name)
    with open(os.path.join(bag_root, "timestamp_info.txt"), "w") as f:
        for topic, method in sorted(timestamp_log.items()):
            f.write(f"{topic}: {method}\n")
@hydra.main(config_path="../../../configs", config_name="data", version_base=None)
def main(cfg: DictConfig) -> None:
    
    input_path = cfg.mcap_path
    if _mode == "list_topics":
        list_topics(input_path)

    elif _mode == "list_transforms":
        list_transforms(input_path)

    else:
        mcaps = [f for f in os.listdir(cfg.bags_path) if f.endswith('.mcap')]
        paths = [os.path.join(cfg.bags_path, mcap) for mcap in mcaps]
        cfg_plain = OmegaConf.to_container(cfg, resolve=True)
        n_proc = min(8, len(paths))
        
        args = [(cfg_plain, p) for p in paths]

        with Pool(processes=n_proc) as pool:
            for i, _ in enumerate(pool.imap_unordered(extract_one, args), start=1):
                print(f'finished with bag {i+1}/{len(paths)}')


if __name__ == "__main__":
    _mode = sys.argv.pop() if len(sys.argv) > 1 else "extract"
    main()