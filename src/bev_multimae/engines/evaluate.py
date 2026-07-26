import os
import logging

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

from bev_multimae.engines.finetune_inference import build_model, move_batch
from bev_multimae.finetuning.centerpoint.decode import decode_centerpoint
# from bev_multimae.datasets.finetuning_data_old import BEVFineData, collate_finetune
from bev_multimae.datasets.finetuning_data import BEVFineData, collate_finetune

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
        "mean_depth_error",
        "mean_lateral_error",
        "mean_euclidean_error",
        "mean_range_error",
        "mean_bearing_error_deg",
        "mean_depth_bias",
        "mean_lateral_bias",
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

def angle_diff(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def axis_angle_diff(a, b):
    d = abs(angle_diff(a, b))
    return min(d, abs(np.pi - d))


def yaw_from_corners(box):
    box = np.asarray(box, dtype=np.float32).squeeze()

    if box.ndim != 2 or box.shape[1] < 2 or box.shape[0] < 4:
        raise ValueError(f"Expected corner box with shape (N, 3), got {box.shape}")

    xy = box[:, :2]
    xy = xy - xy.mean(axis=0, keepdims=True)

    cov = xy.T @ xy

    if not np.isfinite(cov).all() or np.abs(cov).sum() < 1e-8:
        raise ValueError(f"Degenerate corner box: {box}")

    eigvals, eigvecs = np.linalg.eigh(cov)
    main_axis = eigvecs[:, np.argmax(eigvals)]

    return float(np.arctan2(main_axis[1], main_axis[0]))


def box_xyyaw(box):
    box = np.asarray(box, dtype=np.float32).squeeze()

    if box.ndim == 1 and box.shape[0] >= 7:
        return float(box[0]), float(box[1]), float(box[6])

    if box.ndim == 2 and box.shape[1] >= 3 and box.shape[0] >= 4:
        xy = box[:, :2].mean(axis=0)
        yaw = yaw_from_corners(box)
        return float(xy[0]), float(xy[1]), yaw

    raise ValueError(f"Unknown box format: shape={box.shape}, box={box}")


def match_error_values(pred_boxes, target_boxes, matches):
    depth_errs = []
    lateral_errs = []
    euclidean_errs = []
    range_errs = []
    bearing_errs = []

    depth_signed = []
    lateral_signed = []

    for pi, ti, _ in matches:
        px, py, _ = box_xyyaw(pred_boxes[pi])
        tx, ty, _ = box_xyyaw(target_boxes[ti])

        dx = px - tx
        dy = py - ty

        pred_range = np.sqrt(px ** 2 + py ** 2)
        target_range = np.sqrt(tx ** 2 + ty ** 2)

        pred_bearing = np.arctan2(py, px)
        target_bearing = np.arctan2(ty, tx)

        depth_errs.append(abs(dx))
        lateral_errs.append(abs(dy))
        euclidean_errs.append(np.sqrt(dx ** 2 + dy ** 2))
        range_errs.append(abs(pred_range - target_range))
        bearing_errs.append(abs(angle_diff(pred_bearing, target_bearing)))

        depth_signed.append(dx)
        lateral_signed.append(dy)

    return {
        "depth_err": depth_errs,
        "lateral_err": lateral_errs,
        "euclidean_err": euclidean_errs,
        "range_err": range_errs,
        "bearing_err": bearing_errs,
        "depth_bias": depth_signed,
        "lateral_bias": lateral_signed,
    }


def mean_or_nan(values):
    return float(np.mean(values)) if values else float("nan")


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

def drop_inputs(batch, drop_rad=False, drop_feat=False):
    batch = dict(batch)

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

def box_geom(box):
    box = np.asarray(box, dtype=np.float32)

    if box.shape == (8, 3):
        xy = np.ascontiguousarray(box[:, :2], dtype=np.float32)
        poly = cv2.convexHull(xy).reshape(-1, 2)
        return poly, float(box[:, 2].min()), float(box[:, 2].max())

    x, y, z, l, w, h, yaw = box[:7]
    corners = np.array([
        [ l / 2,  w / 2],
        [ l / 2, -w / 2],
        [-l / 2, -w / 2],
        [-l / 2,  w / 2],
    ], dtype=np.float32)

    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    poly = corners @ rot.T + np.array([x, y], dtype=np.float32)

    return poly, float(z - h / 2), float(z + h / 2)


def iou_3d(box_a, box_b):
    poly_a, zmin_a, zmax_a = box_geom(box_a)
    poly_b, zmin_b, zmax_b = box_geom(box_b)

    area_a = cv2.contourArea(poly_a)
    area_b = cv2.contourArea(poly_b)
    inter_area, _ = cv2.intersectConvexConvex(poly_a, poly_b)

    inter_h = max(0.0, min(zmax_a, zmax_b) - max(zmin_a, zmin_b))
    inter = inter_area * inter_h

    vol_a = area_a * (zmax_a - zmin_a)
    vol_b = area_b * (zmax_b - zmin_b)

    return inter / max(vol_a + vol_b - inter, 1e-8)

def iou_bev(box_a, box_b):
    poly_a, _, _ = box_geom(box_a)
    poly_b, _, _ = box_geom(box_b)

    area_a = cv2.contourArea(poly_a)
    area_b = cv2.contourArea(poly_b)
    inter, _ = cv2.intersectConvexConvex(poly_a, poly_b)

    return inter / max(area_a + area_b - inter, 1e-8)


def greedy_match_iou(pred_boxes, pred_scores, target_boxes, iou_thresh=0.5):
    order = np.argsort(-pred_scores)
    matched_targets = set()
    matches = []
    false_pos = []

    for pi in order:
        best_ti = -1
        best_iou = 0.0

        for ti, target in enumerate(target_boxes):
            if ti in matched_targets:
                continue

            iou = iou_3d(pred_boxes[pi], target)

            if iou > best_iou:
                best_iou = iou
                best_ti = ti

        if best_iou >= iou_thresh:
            matched_targets.add(best_ti)
            matches.append((pi, best_ti, best_iou))
        else:
            false_pos.append(pi)

    false_fn = [i for i in range(len(target_boxes)) if i not in matched_targets]
    return matches, false_pos, false_fn

def plot_ap_by_iou(cache, save_dir):
    iou_thresholds = [0.1, 0.2, 0.3, 0.5]
    ap_values = [evaluate_iou_ap(cache, t) for t in iou_thresholds]
    ap_percent = [v * 100 for v in ap_values]

    labels = ["AP@IoU10", "AP@IoU20", "AP@IoU30", "AP@IoU50"]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, ap_percent)
    plt.ylabel("AP [%]")
    plt.title("3D IoU-based AP")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.3)

    for i, value in enumerate(ap_percent):
        plt.text(i, value + 1.0, f"{value:.2f}%", ha="center")

    plt.savefig(
        os.path.join(save_dir, "ap_by_iou.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    return ap_values

def plot_error_curves(results, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    dist_thresholds = sorted(set(r["dist_thresh"] for r in results))
    score_thresholds = sorted(set(r["score_thresh"] for r in results))

    metrics = [
        ("mean_range_error", "Distance error", "Error (m)"),
        ("mean_bearing_error_deg", "Bearing error", "Error (deg)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, (metric, title, ylabel) in zip(axes, metrics):
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
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Match distance",
        loc="upper center",
        ncol=len(dist_thresholds),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(
        os.path.join(save_dir, "localization_error_curves.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def plot_best_error_summary(results, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    valid = [r for r in results if not np.isnan(r["mean_depth_error"])]
    if not valid:
        return

    best = max(valid, key=lambda r: r["f1"])

    names = [
        "Depth",
        "Lateral",
        "Euclidean",
        "Range",
        "Bearing",
    ]
    values = [
        best["mean_depth_error"],
        best["mean_lateral_error"],
        best["mean_euclidean_error"],
        best["mean_range_error"],
        best["mean_bearing_error_deg"],
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values)

    ax.set_title(
        f"Localization errors at best F1 "
        f"(score={best['score_thresh']:.2f}, dist={best['dist_thresh']:.1f} m)"
    )
    ax.set_ylabel("Error (m or deg)")
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        if not np.isnan(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )

    plt.savefig(os.path.join(save_dir, "best_f1_error_summary.png"), dpi=200, bbox_inches="tight")
    plt.close()

def evaluate_iou_ap(cache, iou_thresh=0.5):
    all_scores = []
    all_tp = []
    num_targets = 0

    for item in cache:
        matches, _, _ = greedy_match_iou(
            item["pred_boxes"],
            item["pred_scores"],
            item["target_boxes"],
            iou_thresh,
        )

        matched_ids = {m[0] for m in matches}

        for pi, score in enumerate(item["pred_scores"]):
            all_scores.append(float(score))
            all_tp.append(1 if pi in matched_ids else 0)

        num_targets += len(item["target_boxes"])

    return average_precision(all_scores, all_tp, num_targets)


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

    model_batch = {
        "radar": batch["radar"],
        "bev_feat": batch["bev_feat"],
    }

    model_batch = drop_inputs(
        model_batch,
        drop_rad=cfg.get("drop_rad_inference", False),
        drop_feat=cfg.get("drop_feat_inference", False),
    )

    with torch.no_grad():
        encoder_tokens, _ = model.encoder(model_batch, mask_inputs=False)
        detections = model.detector(encoder_tokens)

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

    for bi, batch in enumerate(loader):
        print(f"batch {bi + 1}/{len(loader)}", flush=True)

        batch = move_batch(batch, device)

        print("  running model", flush=True)
        detections = run_model(model, batch, point_cloud_range, cfg)

        print("  decoding", flush=True)
        decoded = decode(detections, point_cloud_range, cfg, decode_score_thresh)

        print("  saving decoded", flush=True)
        for i, dec in enumerate(decoded):
            pred_boxes = dec["boxes"].detach().cpu().numpy()
            pred_scores = dec["scores"].detach().cpu().numpy()

            pred_xy = pred_boxes[:, :2] if len(pred_boxes) > 0 else np.zeros((0, 2), dtype=np.float32)

            target_boxes = [np.asarray(b) for b in batch["boxes"][i] if b is not None]
            target_xy = np.asarray([box_center_xy(b) for b in target_boxes])

            if target_xy.size == 0:
                target_xy = np.zeros((0, 2), dtype=np.float32)

            cache.append(
                {
                    "pred_xy": pred_xy,
                    "pred_boxes": pred_boxes,
                    "pred_scores": pred_scores,
                    "target_xy": target_xy,
                    "target_boxes": target_boxes,
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

    depth_errs = []
    lateral_errs = []
    euclidean_errs = []
    range_errs = []
    bearing_errs = []
    depth_biases = []
    lateral_biases = []

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
        pred_boxes = item["pred_boxes"][keep]

        matches, false_pos, false_fn = greedy_match(
            pred_xy=pred_xy,
            pred_scores=pred_scores,
            target_xy=target_xy,
            dist_thresh=dist_thresh,
        )

        errs = match_error_values(pred_boxes, item["target_boxes"], matches)

        depth_errs.extend(errs["depth_err"])
        lateral_errs.extend(errs["lateral_err"])
        euclidean_errs.extend(errs["euclidean_err"])
        range_errs.extend(errs["range_err"])
        bearing_errs.extend(errs["bearing_err"])
        depth_biases.extend(errs["depth_bias"])
        lateral_biases.extend(errs["lateral_bias"])

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
        "mean_loc_error": mean_or_nan(loc_errors),
        "mean_depth_error": mean_or_nan(depth_errs),
        "mean_lateral_error": mean_or_nan(lateral_errs),
        "mean_euclidean_error": mean_or_nan(euclidean_errs),
        "mean_range_error": mean_or_nan(range_errs),
        "mean_bearing_error_deg": float(np.rad2deg(mean_or_nan(bearing_errs))) if bearing_errs else float("nan"),
        "mean_depth_bias": mean_or_nan(depth_biases),
        "mean_lateral_bias": mean_or_nan(lateral_biases),
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
    log.info(f"Missing keys: {missing}")
    log.info(f"Unexpected keys: {unexpected}")

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
        f"loc={result['mean_loc_error']:.3f}m | "
        f"depth={result['mean_depth_error']:.3f}m | "
        f"lat={result['mean_lateral_error']:.3f}m | "
        f"euc={result['mean_euclidean_error']:.3f}m | "
        f"bearing={result['mean_bearing_error_deg']:.2f}deg"
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

    img_mean, img_std = None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref_ds, _ = make_dataset(cfg, directions[0], split, img_mean, img_std)
    sample = ref_ds[0]
    if "bev_feat" not in sample:
        raise RuntimeError(
            "bev_feat missing from dataset sample. "
            "You are probably evaluating with a processed dataset that does not contain BEV features."
        )
    model = build_model(cfg, ref_ds.meta)
    model = load_checkpoint(model, cfg)
    model = model.to(device).eval()

    all_cache = []

    for direction in directions:
        ds, point_cloud_range = make_dataset(cfg, direction, split, img_mean, img_std)

        loader = DataLoader(
            ds,
            batch_size=cfg.get("eval_batch_size", 8),
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

        print(f"{direction:>7} | AP@IoU50={evaluate_iou_ap(cache):.3f}")
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

    save_dir = os.path.join(
        cfg.get("plot_folder", "reports/figures"),
        "evaluation",
    )
    os.makedirs(save_dir, exist_ok=True)

    ap_iou10, ap_iou20, ap_iou30, ap_iou50 = plot_ap_by_iou(all_cache, save_dir)

    print(f"\nOverall AP@IoU10={ap_iou10:.3f}")
    print(f"Overall AP@IoU20={ap_iou20:.3f}")
    print(f"Overall AP@IoU30={ap_iou30:.3f}")
    print(f"Overall AP@IoU50={ap_iou50:.3f}")

    ap_iou_path = os.path.join(save_dir, "ap_iou_summary.csv")

    with open(ap_iou_path, "w") as f:
        f.write("iou_thresh,ap\n")
        f.write(f"0.10,{ap_iou10:.6f}\n")
        f.write(f"0.20,{ap_iou20:.6f}\n")
        f.write(f"0.30,{ap_iou30:.6f}\n")
        f.write(f"0.50,{ap_iou50:.6f}\n")

    plot_eval_curves(summary_results, save_dir)
    plot_ap_by_distance(summary_results, save_dir)
    save_eval_table(summary_results, save_dir)

    plot_error_curves(summary_results, save_dir)
    plot_best_error_summary(summary_results, save_dir)  

    print(f"\nSaved evaluation plots and table to: {save_dir}")




if __name__ == "__main__":
    main()