import os
import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt

from bev_multimae.multimae.adapters.rad_adapt import RadarAdapter
from bev_multimae.multimae.adapters.cam_adapt import CameraAdapter
from bev_multimae.multimae.model import Bev_MultiMAE
from bev_multimae.finetuning.centerpoint import (
    TokenToSpatialAdapter,
    CenterPointHead,
    CenterPointDetector,
)
from bev_multimae.datasets.data import BEVDataset, collate_radar


def save_detections(detections, targets, folder):
    os.makedirs(folder, exist_ok=True)

    # Save detection heatmap
    heatmap = detections['heatmap'][0, 0].detach().cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(heatmap, cmap='hot')
    plt.colorbar()
    plt.title('Detection Heatmap')
    plt.axis('off')
    plt.savefig(os.path.join(folder, 'detection_heatmap.png'), bbox_inches='tight', pad_inches=0)
    plt.close()

    # Save target heatmap
    if targets is not None and 'heatmap' in targets:
        target_heatmap = targets['heatmap'][0, 0].detach().cpu().numpy()
        plt.figure(figsize=(8, 8))
        plt.imshow(target_heatmap, cmap='hot')
        plt.colorbar()
        plt.title('Target Heatmap')
        plt.axis('off')
        plt.savefig(os.path.join(folder, 'target_heatmap.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

    # Save comparison
    if targets is not None and 'heatmap' in targets:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(heatmap, cmap='hot')
        axes[0].set_title('Detection')
        axes[0].axis('off')
        axes[1].imshow(target_heatmap, cmap='hot')
        axes[1].set_title('Target')
        axes[1].axis('off')
        plt.savefig(os.path.join(folder, 'heatmap_comparison.png'), bbox_inches='tight', pad_inches=0)
        plt.close()


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    ds = BEVDataset(
        cfg.processed_data_dir_right,
        split="train",
        augment=False,
    )

    meta = ds.meta

    grid_size = meta["grid_size"]
    grid_size_hires = meta["hi_res_grid_size"]

    nx, ny = grid_size[:2]
    nx_hi, ny_hi = grid_size_hires[:2]

    H_cam, W_cam = ny_hi, nx_hi
    patch_size = (H_cam // ny, W_cam // nx)

    dim_tokens = cfg.dim_tokens

    input_adapters = {
        "radar": RadarAdapter(
            dim_tokens,
            grid_size,
            meta["num_point_features"],
            cfg.num_vfe_features,
        ),
        "cam_bev": CameraAdapter(
            dim_tokens,
            cfg.cam_channels,
            patch_size,
            grid_size_hires,
        ),
    }

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_radar,
    )

    # Instantiate encoder
    model = Bev_MultiMAE(
        input_adapters=input_adapters,
        output_adapters=None,
        dim_tokens=dim_tokens,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        drop_path_rate=cfg.drop_path_rate,
        drop_rate=cfg.drop_rate,
        attn_drop_rate=cfg.attn_drop_rate,
    )

    # Instantiate detection head components
    token_adapter = TokenToSpatialAdapter(dim_tokens=dim_tokens, output_channels=128)
    detection_head = CenterPointHead(in_channels=128)
    detector = CenterPointDetector(token_adapter, detection_head)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    detector = detector.to(device).eval()

    batch = next(iter(loader))
    batch["cam_bev"] = batch["cam_bev"].to(device)

    for k, v in batch["radar"].items():
        if isinstance(v, torch.Tensor):
            batch["radar"][k] = v.to(device)

    with torch.no_grad():
        # Get encoder tokens
        encoder_tokens, task_masks = model(
            batch,
            mask_inputs=False,
            num_encoded_tokens=cfg.num_encoded_tokens,
        )

        print("Encoder output:")
        print(f"  encoder_tokens shape: {encoder_tokens.shape}")
        print(f"  task_masks keys: {task_masks.keys()}")

        # Get detections
        detections = detector(encoder_tokens)

        print("\nDetection outputs:")
        for key, val in detections.items():
            print(f"  {key}: {val.shape}")

        # Get targets if available (from finetuning dataset)
        targets = batch.get("detection_targets", None)
        if targets is not None:
            print("\nTarget outputs:")
            for key, val in targets.items():
                print(f"  {key}: {val.shape}")
        else:
            print("\nNo detection targets in batch")

        save_detections(detections, targets, cfg.plot_folder)
        print("\nSaved visualizations to:", cfg.plot_folder)


if __name__ == "__main__":
    main()
