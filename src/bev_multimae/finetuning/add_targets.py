import os
import numpy as np
from pathlib import Path
from targets_utils import build_centerpoint_targets_with_gaussian_gpu, visualize_targets
import hydra

@hydra.main(config_path="../../../configs", config_name="config_finetune", version_base=None)
def main(cfg):
    first_three_box_sample = None

    for direction in ['left', 'right']:
        for dataset in ['train', 'val']:
            path = os.path.join(cfg.finetuning_data_dir, direction, dataset)

            if direction == 'left':
                pcr = cfg.left_point_cloud_range
            else:
                pcr = cfg.right_point_cloud_range

            # Iterate through event folders
            for event_folder in sorted(Path(path).iterdir()):
                print(f"Processing: {direction}, {dataset}, {event_folder}")
                if not event_folder.is_dir():
                    continue

                # Iterate through .npz files in event folder
                for npz_file in sorted(event_folder.glob("*.npz")):

                    # Load the .npz file
                    data = np.load(npz_file, allow_pickle=True)
                    boxes = data['boxes']  # object array, each entry is (8,3) or None

                    # Convert to list, keeping None entries
                    boxes_list = [b for b in boxes]

                    # Build targets from boxes
                    targets = build_centerpoint_targets_with_gaussian_gpu(
                        boxes_list,
                        point_cloud_range=pcr,
                        grid_size=64,
                        gaussian_radius=2,
                        device="cuda",
                    )

                    # Prepare data to save: original boxes + new targets
                    save_dict = {}

                    # Keep all original data
                    for key in data.files:
                        save_dict[key] = data[key]

                    # Add targets with proper naming
                    for target_key, target_tensor in targets.items():
                        save_dict[f'targets_{target_key}'] = target_tensor.cpu().numpy()

                    # Save back to .npz
                    np.savez(npz_file, **save_dict)

                    # Track first sample with 3 boxes
                    num_valid_boxes = sum(1 for b in boxes_list if b is not None)
                    if first_three_box_sample is None and num_valid_boxes == 3:
                        first_three_box_sample = (npz_file, boxes_list, targets, pcr)
                        print(f"  Found first sample with 3 boxes!")

    # Visualize the first sample with 3 boxes
    if first_three_box_sample is not None:
        npz_file, boxes_list, targets, pcr = first_three_box_sample
        vis_path = os.path.join(cfg.finetuning_vis, "target_visualization.png")
        visualize_targets(targets, boxes_list, pcr, vis_path,
                         title=f"Targets for {os.path.basename(npz_file)} (3 boxes)")

if __name__ == "__main__":
    main()