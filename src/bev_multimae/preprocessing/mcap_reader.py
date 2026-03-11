import sys
import os
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

# Hydra
import hydra
from omegaconf import DictConfig
from pathlib import Path
from hydra.utils import to_absolute_path

file_path = 'data/raw/agriRobotData.mcap'

TOPICS = {
    "/sensing/camera/front_left/compressed_image":  "data/raw/camera/front_left",
    "/sensing/camera/front_right/compressed_image": "data/raw/camera/front_right",
    "/sensing/camera/front_left/camera_info":       "data/raw/camera/front_left",
    "/sensing/camera/front_right/camera_info":      "data/raw/camera/front_right",
    "/sensing/radar/front_left/raw/points":         "data/raw/radar/front_left",
    "/sensing/radar/front_right/raw/points":        "data/raw/radar/front_right",
}

def list_topics(input_path):
    """List all unique topics in an MCAP file."""
    with open(input_path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        if summary is None:
            print("No summary available.")
            return
        for channel_id, channel in summary.channels.items():
            schema = summary.schemas.get(channel.schema_id)
            print(f"  {channel.topic:60s}  [{schema.name if schema else 'unknown'}]")


def extract(input_path):
    saved_camera_info = set()  # track which cameras already have their matrix saved

    with open(input_path, "rb") as f:
        # DecoderFactory helps us get real ROS message objects
        reader = make_reader(f, decoder_factories=[DecoderFactory()])

        for schema, channel, message, ros_msg in reader.iter_decoded_messages(topics=list(TOPICS.keys())):
            
            # Output_path
            output_dir = TOPICS[channel.topic]
        
            # Use timestamp for filename so they match
            timestamp = message.log_time
            
            # Image
            if "compressed_image" in channel.topic:
                ext = "jpg"
                
                output_path = os.path.join(output_dir, f"{timestamp}.{ext}")
                with open(output_path, "wb") as img_file:
                    img_file.write(ros_msg.data)
            
            # Camera info — save all calibration data once per camera
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

            # Radar
            elif "points" in channel.topic:
                output_path = os.path.join(output_dir, f"{timestamp}.bin")
                with open(output_path, "wb") as pcd_file:
                    pcd_file.write(ros_msg.data)


@hydra.main(config_path="../../../configs", config_name="data_config", version_base=None)
def main(cfg: DictConfig) -> None:
    input_path = cfg.mcap_path
    if _mode == "list_topics":
        list_topics(input_path)
    else:
        extract(input_path)

if __name__ == "__main__":
    _mode = sys.argv.pop() if len(sys.argv) > 1 else "extract"
    main()