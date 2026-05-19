import hydra
from omegaconf import OmegaConf
import os

from bev_multimae.preprocessing.camera.depth import DepthEstimator
from bev_multimae.finetuning.create_human_boxes import human_boxes_all

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg):
    de = DepthEstimator(cfg, device='cuda', plot=cfg.plotting)
    de._load_model()

    # event = 'evt_0e8QraX8B9UIyxY9'    # evening, two people to the right in the frame
    # event = 'evt_0e8RO9yx2kWoavOD'    # afternoon, person standing still in front of robot while its driving
    # event = 'evt_0e8RSwkcSts5kEaF'    # Two people further away
    # event = 'evt_0e3qa9akdU4BIHaF' 
    event = 'evt_0e8Qmgb6bdukqOwj'      # Lacks distortion coefficients 
    # event = 'evt_0e3rMglACswVRZ1U'    # At night one person
    # event = 'evt_0e8RSh4iCdd5BTkn'

    events = sorted(os.listdir(cfg.mcap_extract_path))

    # events = ['evt_0e8RSwkcSts5kEaF', 'evt_0e8Qmgb6bdukqOwj', 'evt_0e8QraX8B9UIyxY9']
    n_events = len(events)
    direction='left'
    plot = None

    val_events = [
        'evt_0dpi1HrSsDgzvaw2', 'evt_0dyJyd30SYOwBKSH', 'evt_0e8RDkd1DPoqK2oQ', 
        'evt_0e8QmsFL0xUEIEWj', 'evt_0e8RCA7oPH7aLq5j', 'evt_0e3qa9akdU4BIHaF', 
        'evt_0e8QyaRCAyD80b9U', 'evt_0e8Qrf9nfu208uKP', 'evt_0e8Qn5kqXWs3r28T', 
        'evt_0eF19uqU42iRNlag', 'evt_0e3rMOlUlNNs8CD1', 'evt_0e8QoYoT9QEHPW21', 
        'evt_0e3qe7dR7iyLDlAE', 'evt_0e8RRhN10JKbTwss', 'evt_0e8Qmgb6bdukqOwj'
        ]

    save_folder_root = os.path.join(cfg.finetuning_data_dir, direction)

    for event in events:
        OmegaConf.update(cfg, "camera_info", f"data/raw/mcap_extract/{event}/camera/front_{direction}/camera_info.npz")
        OmegaConf.update(cfg, "radar_raw_path", f"data/raw/mcap_extract/{event}/radar/front_{direction}")
        OmegaConf.update(cfg, "imgs_raw_path", f"data/raw/mcap_extract/{event}/camera/front_{direction}")
        OmegaConf.update(cfg, "seg_raw_path", f"data/raw/mcap_extract/{event}/seg/front_{direction}")
        OmegaConf.update(cfg, "bbox_raw_path", f"data/raw/mcap_extract/{event}/bbox/front_{direction}")
        OmegaConf.update(cfg, "lidar_raw_path", f"data/raw/mcap_extract/{event}/lidar/front_top")
        OmegaConf.update(cfg, "mcap_path", f"data/raw/bags/{event}.mcap")
        OmegaConf.update(cfg, "direction", direction)

        save_folder = os.path.join(save_folder_root, 'val' if event in val_events else 'train')

        event_dir = os.path.join(save_folder, event)
        if os.path.exists(event_dir) and os.listdir(event_dir):
            print(f'Skipping {event} — already processed')
            continue

        lidar_path = os.path.join(cfg.mcap_extract_path, event, "lidar", "front_top")
        if not os.path.exists(lidar_path) or not os.listdir(lidar_path):
            print(f'Skipping {event} — empty or missing lidar folder')
            continue

        radar_path = os.path.join(cfg.mcap_extract_path, event, "radar", f"front_{direction}")
        if not os.path.exists(radar_path) or not os.listdir(radar_path):
            print(f'Skipping {event} — empty or missing radar folder')
            continue
        
        camera_info_path = os.path.join(cfg.mcap_extract_path, event, "camera", f"front_{direction}", "camera_info.npz")
        if not os.path.exists(camera_info_path):
            print(f'Skipping {event} — missing camera_info')
            continue
        
        event_boxes = human_boxes_all(cfg, event, de, save_folder)


if __name__ == "__main__":
    main()