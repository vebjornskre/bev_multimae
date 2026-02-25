import sys
import os
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

file_path = 'data/raw/agriRobotData.mcap'

TOPICS = {
    "/sensing/camera/front_left/compressed_image":  "data/raw/camera/front_left",
    "/sensing/camera/front_right/compressed_image": "data/raw/camera/front_right",
    "/sensing/radar/front_left/raw/points":         "data/raw/radar/front_left",
    "/sensing/radar/front_right/raw/points":        "data/raw/radar/front_right",
}

def extract(input_path):
    with open(input_path, "rb") as f:
        # DecoderFactory helps us get real ROS message objects
        reader = make_reader(f, decoder_factories=[DecoderFactory()])

        for schema, channel, message, ros_msg in reader.iter_decoded_messages(topics=list(TOPICS.keys())):
            
            # 1. Determine output path
            output_dir = TOPICS[channel.topic]
            os.makedirs(output_dir, exist_ok=True)
            
            # Use timestamp for filename so they match
            timestamp = message.log_time
            
            # 2. Extract Image
            if "compressed_image" in channel.topic:
                # ros_msg.format usually strings like "jpeg" or "png"
                ext = "png" if "png" in ros_msg.format else "jpg"
                
                output_path = os.path.join(output_dir, f"{timestamp}.{ext}")
                with open(output_path, "wb") as img_file:
                    img_file.write(ros_msg.data)
            
            # 3. Extract Radar (PointCloud2)
            elif "points" in channel.topic:
                # Save as numpy .npy files (easiest to load later)
                # The data is a raw byte buffer. 
                # To make it usable, we usually view it as float32.
                
                # Check point_step (bytes per point) to guess structure if unknown
                # Common: 32 bytes (x, y, z, intensity, ring, ...) or 16 bytes (x, y, z, i)
                
                # For now, let's just save the raw byte buffer to a .bin file 
                # This mimics the KITTI format.
                output_path = os.path.join(output_dir, f"{timestamp}.bin")
                with open(output_path, "wb") as pcd_file:
                    pcd_file.write(ros_msg.data)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python mcap_reader.py <path_to_mcap>")
        sys.exit(1)
        
    extract(input_path=sys.argv[1])