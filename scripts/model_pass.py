import os
import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt

from bev_multimae.multimae.adapters.rad_adapt import RadarAdapter
from bev_multimae.multimae.adapters.cam_adapt import CameraAdapter
from bev_multimae.multimae.decoders.recon_decoder import SpatialOutputAdapter
from bev_multimae.multimae.model import Bev_MultiMAE
from bev_multimae.datasets.data import BEVDataset, collate_radar


def save_preds(preds, folder):
    os.makedirs(folder, exist_ok=True)

    for k, v in preds.items():
        x = v[0].detach().cpu()

        if x.dim() == 3:
            x = x.permute(1, 2, 0)

        x = x.numpy()
        x = (x - x.min()) / (x.max() - x.min() + 1e-6)

        plt.figure()
        if k == "radar":
            plt.imshow(x[..., 0], cmap="gray")
        else:
            plt.imshow(x[..., :3])

        plt.axis("off")
        plt.savefig(os.path.join(folder, f"{k}.png"), bbox_inches="tight", pad_inches=0)
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

    output_adapters = {
        "cam_bev": SpatialOutputAdapter(
            num_channels=cfg.cam_channels,
            stride_level=1,
            patch_size_full=patch_size,
            image_size=(grid_size_hires[1], grid_size_hires[0]),
            task="cam_bev",
            context_tasks=["cam_bev", "radar"],
            dim_tokens=dim_tokens,
            dim_tokens_enc=dim_tokens,
        ),
        "radar": SpatialOutputAdapter(
            num_channels=cfg.rad_channels,
            stride_level=1,
            patch_size_full=(1, 1),
            image_size=(grid_size[1], grid_size[0]),
            task="radar",
            context_tasks=["cam_bev", "radar"],
            dim_tokens_enc=dim_tokens,
        ),
    }

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_radar,
    )

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    batch = next(iter(loader))
    batch["cam_bev"] = batch["cam_bev"].to(device)

    for k, v in batch["radar"].items():
        if isinstance(v, torch.Tensor):
            batch["radar"][k] = v.to(device)

    with torch.no_grad():
        encoder_tokens, task_masks = model(
            batch,
            mask_inputs=False,
            num_encoded_tokens=cfg.num_encoded_tokens,
        )

        print(type(encoder_tokens))
        print(encoder_tokens.shape)
        print(task_masks.keys())

    # save_preds(preds, cfg.plot_folder)
    # print("saved to:", cfg.plot_folder)


if __name__ == "__main__":
    main()