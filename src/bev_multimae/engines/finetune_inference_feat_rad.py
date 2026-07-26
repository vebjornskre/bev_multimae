import os
import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
from time import perf_counter

from bev_multimae.datasets.finetuning_data import BEVFineData
from bev_multimae.engines.finetune import collate_finetune
from bev_multimae.finetuning.centerpoint.decode import decode_centerpoint, apply_double_flip_augmentation
from bev_multimae.finetuning.centerpoint.model import CenterPointDetector, CenterPointHead
from bev_multimae.finetuning.centerpoint.token_adapter import TokenToSpatialAdapter
from bev_multimae.finetuning.model_lightning import CenterPointLightning
from bev_multimae.multimae.adapters.rad_adapt import RadarAdapter
from bev_multimae.multimae.adapters.feat_adapt import FeatureAdapter
from bev_multimae.multimae.model import Bev_MultiMAE
from bev_multimae.preprocessing.get_transforms import apply_transform
from bev_multimae.visualization.finetuning_predictions import save_detections, plot_boxes_on_image


def save_ablation_heatmaps(ablation, folder, max_pool=False):
    
    os.makedirs(folder, exist_ok=True)

    hms = {}
    vmax = 0.1

    for name, item in ablation.items():
        det = item["detections"]
        hm = det["heatmap"].sigmoid()
        if max_pool:
            hm = torch.nn.functional.max_pool2d(hm, kernel_size=3, stride=1, padding=1).eq(hm) * hm
        hm = hm[0, 0].detach().cpu().numpy()
        hms[name] = hm
        vmax = max(vmax, hm.max())

    fig, axes = plt.subplots(1, len(hms), figsize=(6 * len(hms), 5))
    fig.subplots_adjust(right=0.92, wspace=0.08)

    if len(hms) == 1:
        axes = [axes]

    for ax, (name, hm) in zip(axes, hms.items()):
        im = ax.imshow(hm, cmap="inferno", vmin=0, vmax=vmax, origin="lower")
        ax.set_title(name)
        ax.axis("off")

    cax = fig.add_axes([0.94, 0.18, 0.015, 0.64])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Heatmap probability")

    plt.savefig(os.path.join(folder, "ablation_heatmaps.png"), bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close()

def to_cam(boxes, T_ego_cam):
    return [apply_transform(T_ego_cam, b) for b in boxes]


def box_to_corners(box):
    x, y, z, l, w, h, yaw = box

    dx, dy, dz = l / 2, w / 2, h / 2

    corners = torch.tensor([
        [ dx,  dy, -dz],
        [ dx, -dy, -dz],
        [-dx, -dy, -dz],
        [-dx,  dy, -dz],
        [ dx,  dy,  dz],
        [ dx, -dy,  dz],
        [-dx, -dy,  dz],
        [-dx,  dy,  dz],
    ], dtype=box.dtype, device=box.device)

    c, s = torch.cos(yaw), torch.sin(yaw)

    rot = torch.tensor([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ], dtype=box.dtype, device=box.device)

    return corners @ rot.T + torch.tensor([x, y, z], dtype=box.dtype, device=box.device)


def print_ablation_stats(batch):
    print("== Input stats ==")

    for k in ["cam_bev", "bev_feat"]:
        if k in batch:
            x = batch[k].detach()
            print(
                f"{k}: shape={tuple(x.shape)}, "
                f"mean={x.mean().item():.4f}, std={x.std().item():.4f}, "
                f"min={x.min().item():.4f}, max={x.max().item():.4f}"
            )

    if "radar" in batch and "points" in batch["radar"]:
        pts = batch["radar"]["points"].detach()
        vals = pts[:, 1:] if pts.shape[1] > 1 else pts
        print(
            f"radar.points: shape={tuple(pts.shape)}, "
            f"mean={vals.mean().item():.4f}, std={vals.std().item():.4f}, "
            f"min={vals.min().item():.4f}, max={vals.max().item():.4f}"
        )

    for k in ["f_cluster", "f_center"]:
        if "radar" in batch and k in batch["radar"]:
            x = batch["radar"][k].detach()
            print(
                f"radar.{k}: shape={tuple(x.shape)}, "
                f"mean={x.mean().item():.4f}, std={x.std().item():.4f}, "
                f"min={x.min().item():.4f}, max={x.max().item():.4f}"
            )

def move_batch(batch, device):
    if "cam_bev" in batch:
        batch["cam_bev"] = batch["cam_bev"].to(device)

    if "bev_feat" in batch:
        batch["bev_feat"] = batch["bev_feat"].to(device)

    for k, v in batch["radar"].items():
        if isinstance(v, torch.Tensor):
            batch["radar"][k] = v.to(device)

    if "targets" in batch:
        for k, v in batch["targets"].items():
            batch["targets"][k] = v.to(device)

    for k in ["img_2d", "K", "D", "T_cam_ego"]:
        if k in batch:
            batch[k] = batch[k].to(device)

    return batch


def build_model(cfg, meta):
    grid_size = meta["grid_size"]
    nx, ny = grid_size[:2]

    dim_tokens = cfg.dim_tokens

    bev_feat_grid_size = meta.get("bev_feat_grid_size", cfg.bev_feat_grid_size)
    bev_feat_channels = cfg.bev_feat_channels

    nx_feat, ny_feat = bev_feat_grid_size[:2]
    feat_patch_size = (ny_feat // ny, nx_feat // nx)

    if ny_feat % ny != 0 or nx_feat % nx != 0:
        raise ValueError(
            f"bev_feat_grid_size {bev_feat_grid_size} must be divisible by token grid {grid_size}"
        )

    input_adapters = {
        "radar": RadarAdapter(
            dim_tokens,
            grid_size,
            meta["num_point_features"],
            cfg.num_vfe_features,
        ),
        "bev_feat": FeatureAdapter(
            d_model=dim_tokens,
            channels=bev_feat_channels,
            patch_size=feat_patch_size,
            bev_feat_grid_size=(ny_feat, nx_feat),
        ),
    }

    encoder = Bev_MultiMAE(
        input_adapters=input_adapters,
        output_adapters=None,
        dim_tokens=dim_tokens,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        drop_path_rate=cfg.drop_path_rate,
        drop_rate=cfg.drop_rate,
        attn_drop_rate=cfg.attn_drop_rate,
    )

    token_adapter = TokenToSpatialAdapter(
        dim_tokens=dim_tokens,
        output_channels=cfg.centerpoint_channels,
        include_global=cfg.include_global_token,
    )

    detector = CenterPointDetector(
        token_adapter,
        CenterPointHead(
            in_channels=cfg.centerpoint_channels,
            num_backbone_layers=cfg.get("num_backbone_layers", 2),
            dropout=cfg.get("centerpoint_dropout", 0.0),
        ),
    )

    return CenterPointLightning(
        encoder=encoder,
        detector=detector,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        heatmap_weight=cfg.heatmap_weight,
        offset_weight=cfg.offset_weight,
        height_weight=cfg.height_weight,
        dim_weight=cfg.dim_weight,
        rot_weight=cfg.rot_weight,
    )

def run_detect(model, batch, point_cloud_range, cfg):
    if "bev_feat" not in batch:
        raise RuntimeError("bev_feat is missing from batch")

    model_batch = {
        "radar": batch["radar"],
        "bev_feat": batch["bev_feat"],
    }

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = perf_counter()

    with torch.no_grad():
        encoder_tokens, _ = model.encoder(model_batch, mask_inputs=False)
        detections = model.detector(encoder_tokens)

        if cfg.get("use_double_flip", False):
            detections = apply_double_flip_augmentation(detections, point_cloud_range)

    decoded = decode_centerpoint(
        detections,
        point_cloud_range=point_cloud_range,
        score_thresh=cfg.get("score_thresh", 0.1),
        post_center_range=point_cloud_range,
        topk=cfg.get("topk", 20),
        use_circle_nms=cfg.get("use_circle_nms", True),
        min_radius=cfg.get("min_radius", 0.3),
        nms_post_max_size=cfg.get("nms_post_max_size", 50),
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = perf_counter() - t0
    print(f"[PROFILE] Inference + decode time: {dt:.4f} s")

    return detections, decoded


def setup_and_infer(cfg, sample_idx, visualize=True):
    direction = cfg.get("direction", "right")
    split = cfg.get("split", "val")

    pretrain_path = cfg.processed_data_dir_right if direction == "right" else cfg.processed_data_dir_left
    point_cloud_range = cfg.right_point_cloud_range if direction == "right" else cfg.left_point_cloud_range

    try:
        ms = torch.load(os.path.join(cfg.processed_data_dir, "mean_std.pt"), map_location="cpu")
        img_mean, img_std = ms["img_mean"], ms["img_std"]
    except Exception:
        img_mean, img_std = None, None

    ds = BEVFineData(
        pretrain_path=pretrain_path,
        finetune_path=cfg.finetuning_data_dir,
        direction=direction,
        split=split,
        img_mean=img_mean,
        img_std=img_std,
        point_cloud_range=point_cloud_range,
        augment=False,
        img_2d=True,
    )

    if sample_idx < 0 or sample_idx >= len(ds):
        raise IndexError(f"sample_idx {sample_idx} outside dataset length {len(ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg, ds.meta)
    ckpt_path = os.path.join("fine_model_checkpoints", cfg.get("best_model"))
    ckpt = torch.load(ckpt_path, map_location="cpu")

    sd = ckpt["state_dict"]
    sd = {k.replace("detector._orig_mod.", "detector."): v for k, v in sd.items()}

    model.load_state_dict(sd, strict=True)
    model = model.to(device)

    for m in model.encoder.input_adapters.values():
        m.to(device)

    model = model.eval()

    batch = collate_finetune([ds[sample_idx]])
    batch = move_batch(batch, device)

    if "bev_feat" not in batch:
        raise RuntimeError(
            "bev_feat missing from batch. Check that processed_data_dir_right/left points to data with BEV features."
        )

    if cfg.get("drop_rad_inference", False):
        batch["radar"]["points"][:, 1:] = 0

        if "f_cluster" in batch["radar"]:
            batch["radar"]["f_cluster"] = torch.zeros_like(batch["radar"]["f_cluster"])

        if "f_center" in batch["radar"]:
            batch["radar"]["f_center"] = torch.zeros_like(batch["radar"]["f_center"])


    if cfg.get("drop_feat_inference", False) and "bev_feat" in batch:
        batch["bev_feat"] = torch.zeros_like(batch["bev_feat"])

    if cfg.get("ablation_test", False):
        print_ablation_stats(batch)

    detections, decoded = run_detect(model, batch, point_cloud_range, cfg)
    ablation = {}

    if cfg.get("ablation_test", False):
        ablation = {
            "real": {"detections": detections, "decoded": decoded},
        }

        batch_no_rad = copy.deepcopy(batch)
        batch_no_rad["radar"]["points"][:, 1:] = 0

        if "f_cluster" in batch_no_rad["radar"]:
            batch_no_rad["radar"]["f_cluster"] = torch.zeros_like(batch_no_rad["radar"]["f_cluster"])

        if "f_center" in batch_no_rad["radar"]:
            batch_no_rad["radar"]["f_center"] = torch.zeros_like(batch_no_rad["radar"]["f_center"])

        det_no_rad, dec_no_rad = run_detect(model, batch_no_rad, point_cloud_range, cfg)
        ablation["no_radar"] = {"detections": det_no_rad, "decoded": dec_no_rad}

        batch_no_feat = copy.deepcopy(batch)
        batch_no_feat["bev_feat"] = torch.zeros_like(batch_no_feat["bev_feat"])
        det_no_feat, dec_no_feat = run_detect(model, batch_no_feat, point_cloud_range, cfg)
        ablation["no_bev_feat"] = {"detections": det_no_feat, "decoded": dec_no_feat}

        print("== Ablation test ==")
        for name, item in ablation.items():
            dec = item["decoded"]
            scores = dec[0]["scores"]
            boxes = dec[0]["boxes"]
            top_score = scores.max().item() if scores.numel() else 0.0
            print(f"{name}: n={len(boxes)}, top_score={top_score:.3f}")

    boxes = decoded[0]["boxes"]
    pred_boxes = [box_to_corners(b).detach().cpu().numpy() for b in boxes]

    target_boxes = []
    for b in batch["boxes"][0]:
        if b is not None:
            if torch.is_tensor(b):
                b = b.detach().cpu().numpy()
            target_boxes.append(b)

    targets = batch.get("targets", None)

    if visualize and "img_2d" in batch:
        save_path = os.path.join(
            cfg.plot_folder,
            "finetuning",
            "predictions",
            str(sample_idx),
        )
        os.makedirs(save_path, exist_ok=True)
        save_ablation_heatmaps(ablation, save_path, max_pool=cfg.max_pool)

        save_detections(
            detections,
            targets,
            batch,
            save_path,
            pred_boxes=pred_boxes,
            target_boxes=target_boxes,
            point_cloud_range=point_cloud_range,
            voxel_size=cfg.hi_res_voxel,
        )

        T_cam_ego = batch["T_cam_ego"][0].detach().cpu().numpy()
        T_ego_cam = np.linalg.inv(T_cam_ego)

        plot_boxes_on_image(
            batch["img_2d"],
            to_cam(pred_boxes, T_ego_cam),
            to_cam(target_boxes, T_ego_cam),
            batch["K"],
            batch["D"],
            save_path=os.path.join(save_path, "boxes_on_image.png"),
        )

    return detections, targets, decoded, ablation