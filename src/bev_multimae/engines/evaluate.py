import os
import logging

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from bev_multimae.datasets.finetuning_data import BEVFineData, collate_finetune
from bev_multimae.engines.finetune_inference import build_model, move_batch
from bev_multimae.finetuning.centerpoint.decode import decode_centerpoint, apply_double_flip_augmentation

log = logging.getLogger(__name__)

def plot_eval_curves(results, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    dist_thresholds = sorted(set(r["dist_thresh"] for r in results))
    score_thresholds = sorted(set(r["score_thresh"] for r in results))

    metrics = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("fp_per_frame", "False positives / frame"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, (metric, title) in zip(axes, metrics):
        for dist in dist_thresholds:
            vals = []
            for score in score_thresholds:
                match = [
                    r for r in results
                    if r["score_thresh"] == score and r["dist_thresh"] == dist
                ]
                vals.append(match[0][metric] if match else np.nan)

            ax.plot(score_thresholds, vals, marker="o", label=f"{dist:.1f} m")

        ax.set_title(title)
        ax.set_xlabel("Score threshold")
        ax.grid(True, alpha=0.3)

        if metric != "fp_per_frame":
            ax.set_ylim(0, 1)

    axes[0].set_ylabel("Metric value")
    axes[2].set_ylabel("Metric value")
    axes[3].set_ylabel("FP/frame")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Match distance", loc="upper center", ncol=len(dist_thresholds))

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(save_dir, "evaluation_curves.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_ap_by_distance(results, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    dist_thresholds = sorted(set(r["dist_thresh"] for r in results))
    ap_values = []

    for dist in dist_thresholds:
        vals = [r["ap"] for r in results if r["dist_thresh"] == dist]
        ap_values.append(vals[0] if vals else np.nan)

    plt.figure(figsize=(7, 5))
    plt.plot(dist_thresholds, ap_values, marker="o")
    plt.xlabel("Center-distance threshold (m)")
    plt.ylabel("AP")
    plt.title("Center-distance AP")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "ap_by_distance.png"), dpi=200, bbox_inches="tight")
    plt.close()


def save_eval_table(results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "evaluation_summary.csv")

    keys = [
        "score_thresh",
        "dist_thresh",
        "ap",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "num_preds",
        "num_targets",
        "num_frames",
        "fp_per_frame",
        "mean_loc_error",
    ]

    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in results:
            f.write(",".join(str(r[k]) for k in keys) + "\n")


def box_center_xy(box):
    if torch.is_tensor(box):
        box = box.detach().cpu().numpy()

    box = np.asarray(box)

    if box.shape == (8, 3):
        return box.mean(axis=0)[:2]

    if box.shape[-1] >= 2:
        return box[:2]

    raise ValueError(f"Unknown box shape: {box.shape}")


def greedy_match(pred_xy, pred_scores, target_xy, dist_thresh):
    if len(pred_xy) == 0:
        return [], [], list(range(len(target_xy)))

    order = np.argsort(-pred_scores)
    matched_targets = set()
    matches = []
    false_pos = []

    for pi in order:
        if len(target_xy) == 0:
            false_pos.append(pi)
            continue

        dists = np.linalg.norm(target_xy - pred_xy[pi][None, :], axis=1)
        ti = int(np.argmin(dists))

        if dists[ti] <= dist_thresh and ti not in matched_targets:
            matched_targets.add(ti)
            matches.append((pi, ti, float(dists[ti])))
        else:
            false_pos.append(pi)

    false_fn = [i for i in range(len(target_xy)) if i not in matched_targets]
    return matches, false_pos, false_fn

def drop_inputs(batch, drop_cam=False, drop_rad=False, drop_feat=False):
    if drop_cam:
        if "cam_bev" in batch:
            batch["cam_bev"] = torch.zeros_like(batch["cam_bev"])
        if "img_2d" in batch and torch.is_tensor(batch["img_2d"]):
            batch["img_2d"] = torch.zeros_like(batch["img_2d"])

    if drop_rad and "radar" in batch:
        radar = dict(batch["radar"])

        if "points" in radar:
            radar["points"] = radar["points"].clone()
            radar["points"][:, 1:] = 0

        if "f_cluster" in radar:
            radar["f_cluster"] = torch.zeros_like(radar["f_cluster"])

        if "f_center" in radar:
            radar["f_center"] = torch.zeros_like(radar["f_center"])

        batch["radar"] = radar

    if drop_feat and "bev_feat" in batch:
        batch["bev_feat"] = torch.zeros_like(batch["bev_feat"])

    return batch


def average_precision(all_scores, all_tp, num_targets):
    if len(all_scores) == 0 or num_targets == 0:
        return 0.0

    order = np.argsort(-np.asarray(all_scores))
    tp = np.asarray(all_tp)[order].astype(np.float32)
    fp = 1.0 - tp

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recall = cum_tp / max(num_targets, 1)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)

    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def run_model(model, batch, point_cloud_range, cfg):
    batch = drop_inputs(
        batch,
        drop_cam=cfg.get("drop_cam_inference", False),
        drop_rad=cfg.get("drop_rad", False),
        drop_feat=cfg.get("drop_feat", False),
    )

    print(f"drop_cam_inference: {cfg.get('drop_cam_inference', False)}")
    print(f"drop_rad: {cfg.get('drop_rad', False)}")
    print(f"drop_feat: {cfg.get('drop_feat', False)}")

    with torch.no_grad():
        encoder_tokens, _ = model.encoder(batch, mask_inputs=False)
        detections = model.detector(encoder_tokens)

        if cfg.get("use_double_flip", False):
            detections = apply_double_flip_augmentation(detections, point_cloud_range)

    return detections


def decode(detections, point_cloud_range, cfg, decode_score_thresh):
    return decode_centerpoint(
        detections,
        point_cloud_range=point_cloud_range,
        score_thresh=decode_score_thresh,
        post_center_range=point_cloud_range,
        topk=cfg.get("topk", 100),
        use_circle_nms=cfg.get("use_circle_nms", True),
        min_radius=cfg.get("min_radius", 0.3),
        nms_post_max_size=cfg.get("nms_post_max_size", 100),
    )


def collect_predictions(model, loader, device, point_cloud_range, cfg, decode_score_thresh):
    cache = []

    for batch in loader:
        batch = move_batch(batch, device)

        detections = run_model(model, batch, point_cloud_range, cfg)
        decoded = decode(detections, point_cloud_range, cfg, decode_score_thresh)

        for i, dec in enumerate(decoded):
            pred_boxes = dec["boxes"].detach().cpu().numpy()
            pred_scores = dec["scores"].detach().cpu().numpy()

            if len(pred_boxes) > 0:
                pred_xy = pred_boxes[:, :2]
            else:
                pred_xy = np.zeros((0, 2), dtype=np.float32)

            target_boxes = batch["boxes"][i]
            target_xy = np.asarray([box_center_xy(b) for b in target_boxes if b is not None])

            if target_xy.size == 0:
                target_xy = np.zeros((0, 2), dtype=np.float32)

            cache.append(
                {
                    "pred_xy": pred_xy,
                    "pred_scores": pred_scores,
                    "target_xy": target_xy,
                }
            )

    return cache


def evaluate_cache(cache, dist_thresh, eval_score_thresh, decode_score_thresh):
    all_scores = []
    all_tp = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_targets = 0
    total_preds = 0
    loc_errors = []

    for item in cache:
        pred_xy_all = item["pred_xy"]
        pred_scores_all = item["pred_scores"]
        target_xy = item["target_xy"]

        ap_matches, _, _ = greedy_match(
            pred_xy=pred_xy_all,
            pred_scores=pred_scores_all,
            target_xy=target_xy,
            dist_thresh=dist_thresh,
        )
        ap_matched_pred_ids = {m[0] for m in ap_matches}

        for pi, score in enumerate(pred_scores_all):
            all_scores.append(float(score))
            all_tp.append(1 if pi in ap_matched_pred_ids else 0)

        keep = pred_scores_all >= eval_score_thresh
        pred_xy = pred_xy_all[keep]
        pred_scores = pred_scores_all[keep]

        matches, false_pos, false_fn = greedy_match(
            pred_xy=pred_xy,
            pred_scores=pred_scores,
            target_xy=target_xy,
            dist_thresh=dist_thresh,
        )

        total_tp += len(matches)
        total_fp += len(false_pos)
        total_fn += len(false_fn)
        total_targets += len(target_xy)
        total_preds += len(pred_xy)
        loc_errors.extend([m[2] for m in matches])

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    ap = average_precision(all_scores, all_tp, total_targets)
    fp_per_frame = total_fp / max(len(cache), 1)
    mean_loc_error = float(np.mean(loc_errors)) if loc_errors else float("nan")

    return {
        "score_thresh": eval_score_thresh,
        "decode_score_thresh": decode_score_thresh,
        "dist_thresh": dist_thresh,
        "ap": ap,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "num_preds": total_preds,
        "num_targets": total_targets,
        "num_frames": len(cache),
        "fp_per_frame": fp_per_frame,
        "mean_loc_error": mean_loc_error,
    }


def make_dataset(cfg, direction, split, img_mean, img_std):
    pretrain_path = cfg.processed_data_dir_right if direction == "right" else cfg.processed_data_dir_left
    point_cloud_range = cfg.right_point_cloud_range if direction == "right" else cfg.left_point_cloud_range

    ds = BEVFineData(
        pretrain_path=pretrain_path,
        finetune_path=cfg.finetuning_data_dir,
        direction=direction,
        split=split,
        img_mean=img_mean,
        img_std=img_std,
        point_cloud_range=point_cloud_range,
        augment=False,
        img_2d=False,
    )

    return ds, point_cloud_range


def load_norm_stats(cfg):
    try:
        ms = torch.load(os.path.join(cfg.processed_data_dir, "mean_std.pt"), map_location="cpu")
        return ms["img_mean"], ms["img_std"]
    except Exception:
        log.warning("Could not load normalization stats, using None")
        return None, None


def load_checkpoint(model, cfg):
    ckpt_path = os.path.join(cfg.fine_model_folder, cfg.get("best_model"))
    ckpt = torch.load(ckpt_path, map_location="cpu")

    state_dict = ckpt["state_dict"]
    state_dict = {k.replace("detector._orig_mod.", "detector."): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    log.info(f"Loaded checkpoint: {ckpt_path}")
    log.info(f"Missing keys: {len(missing)}")
    log.info(f"Unexpected keys: {len(unexpected)}")

    return model


def print_result(name, result):
    print(
        f"{name:>7} | "
        f"score={result['score_thresh']:.2f} | "
        f"dist={result['dist_thresh']:.2f}m | "
        f"AP={result['ap']:.3f} | "
        f"P={result['precision']:.3f} | "
        f"R={result['recall']:.3f} | "
        f"F1={result['f1']:.3f} | "
        f"TP={result['tp']} FP={result['fp']} FN={result['fn']} | "
        f"preds={result['num_preds']} targets={result['num_targets']} | "
        f"FP/frame={result['fp_per_frame']:.3f} | "
        f"loc_err={result['mean_loc_error']:.3f}m"
    )


def sweep_cache(name, cache, score_thresholds, dist_thresholds, decode_score_thresh):
    results = []

    for score_thresh in score_thresholds:
        for dist_thresh in dist_thresholds:
            result = evaluate_cache(
                cache=cache,
                dist_thresh=float(dist_thresh),
                eval_score_thresh=float(score_thresh),
                decode_score_thresh=decode_score_thresh,
            )
            results.append(result)
            print_result(name, result)

    return results

@hydra.main(config_path="../../../configs", config_name="config_finetune", version_base=None)
def main(cfg: DictConfig):
    split = cfg.get("eval_split", cfg.get("split", "val"))
    directions = list(cfg.get("eval_directions", ["right", "left"]))
    score_thresholds = list(cfg.get("score_thresholds", [0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]))
    dist_thresholds = list(cfg.get("center_dist_thresholds", [0.5, 1.0, 2.0]))
    decode_score_thresh = float(cfg.get("decode_score_thresh", min(score_thresholds)))

    print(f"Evaluating checkpoint: {cfg.get('best_model')}")
    print(f"split: {split}")
    print(f"directions: {directions}")
    print(f"score thresholds: {score_thresholds}")
    print(f"center distance thresholds: {dist_thresholds}")
    print(f"decode score threshold: {decode_score_thresh}")
    print(f"use_double_flip: {cfg.get('use_double_flip', False)}")

    img_mean, img_std = load_norm_stats(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref_ds, _ = make_dataset(cfg, directions[0], split, img_mean, img_std)
    model = build_model(cfg, ref_ds.meta)
    model = load_checkpoint(model, cfg)
    model = model.to(device).eval()

    all_cache = []

    for direction in directions:
        ds, point_cloud_range = make_dataset(cfg, direction, split, img_mean, img_std)

        loader = DataLoader(
            ds,
            batch_size=cfg.get("eval_batch_size", cfg.batch_size),
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=True if cfg.num_workers > 0 else False,
            collate_fn=collate_finetune,
        )

        print(f"\nCaching predictions for {direction} / {split}: {len(ds)} samples")

        cache = collect_predictions(
            model=model,
            loader=loader,
            device=device,
            point_cloud_range=point_cloud_range,
            cfg=cfg,
            decode_score_thresh=decode_score_thresh,
        )

        all_cache.extend(cache)

        print(f"\nEvaluating {direction} / {split}")
        sweep_cache(
            name=direction,
            cache=cache,
            score_thresholds=score_thresholds,
            dist_thresholds=dist_thresholds,
            decode_score_thresh=decode_score_thresh,
        )

    print("\nSummary")
    summary_results = sweep_cache(
        name="all",
        cache=all_cache,
        score_thresholds=score_thresholds,
        dist_thresholds=dist_thresholds,
        decode_score_thresh=decode_score_thresh,
    )

    save_dir = os.path.join(cfg.get("plot_folder", "reports/figures"), "evaluation")
    plot_eval_curves(summary_results, save_dir)
    plot_ap_by_distance(summary_results, save_dir)
    save_eval_table(summary_results, save_dir)

    print(f"\nSaved evaluation plots and table to: {save_dir}")


if __name__ == "__main__":
    main()