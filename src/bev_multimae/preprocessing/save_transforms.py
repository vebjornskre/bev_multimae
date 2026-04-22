import numpy as np
from bev_multimae.preprocessing.mcap_reader import list_transforms, list_topics
import os
import hydra
from mcap.reader import make_reader
from datetime import datetime, timezone


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg):

    folder = cfg.bags_path

    for mcap in os.listdir(folder):

        with open(os.path.join(folder, mcap), "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()
            if summary is None:
                print("No summary available.")
                return
        

            if any("tf_static" in ch.topic for ch in summary.channels.values()):
                print(f"TF_Static in {mcap}")

                for schema, channel, message in reader.iter_messages():
                    if "tf_static" in channel.topic:
                        ts_ns = message.log_time
                        ts_s = ts_ns / 1e9
                        dt = datetime.fromtimestamp(ts_s, tz=timezone.utc)
                        print(dt.strftime("%Y-%m-%d %H:%M:%S UTC"))
                        break

                print()
        
if __name__ == '__main__':
    main()

