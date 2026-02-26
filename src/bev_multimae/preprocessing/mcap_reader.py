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
            
            # Output_path
            output_dir = TOPICS[channel.topic]
        
            # Use timestamp for filename so they match
            timestamp = message.log_time
            
            # Image
            if "compressed_image" in channel.topic:
                # ros_msg.format usually strings like "jpeg" or "png"
                ext = "png" if "png" in ros_msg.format else "jpg"
                
                output_path = os.path.join(output_dir, f"{timestamp}.{ext}")
                with open(output_path, "wb") as img_file:
                    img_file.write(ros_msg.data)
            
            # Radar
            elif "points" in channel.topic:
                output_path = os.path.join(output_dir, f"{timestamp}.bin")
                with open(output_path, "wb") as pcd_file:
                    pcd_file.write(ros_msg.data)

                # Debug prints
                print("Topic:", channel.topic)
                print("Fields:")
                for f in ros_msg.fields:
                    print("  name:", f.name,
                        "| datatype:", f.datatype,
                        "| offset:", f.offset)

                print("Point step:", ros_msg.point_step)
                print("Data length:", len(ros_msg.data))

                break
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python mcap_reader.py <path_to_mcap>")
        sys.exit(1)
        
    extract(input_path=sys.argv[1])