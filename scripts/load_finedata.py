import os
import torch
import hydra
from omegaconf import DictConfig

from bev_multimae.datasets.finetuning_data import BEVFineData


def load_norm(cfg):
    try:
        ms = torch.load(os.path.join(cfg.processed_data_dir, "mean_std.pt"), map_location="cpu")
        return ms["img_mean"], ms["img_std"]
    except Exception:
        return None, None


def make_ds(cfg, direction, split, img_mean, img_std):
    if direction == "right":
        pretrain_path = cfg.processed_data_dir_right
        point_cloud_range = cfg.right_point_cloud_range
    else:
        pretrain_path = cfg.processed_data_dir_left
        point_cloud_range = cfg.left_point_cloud_range

    return BEVFineData(
        pretrain_path=pretrain_path,
        finetune_path=cfg.finetuning_data_dir,
        direction=direction,
        split=split,
        img_mean=img_mean,
        img_std=img_std,
        point_cloud_range=point_cloud_range,
        augment=False,
    )


def num_persons(sample):
    targets = sample.get("targets", {})

    for key in ["mask", "reg_mask", "obj_mask", "valid_mask"]:
        if key in targets:
            return int(targets[key].sum().item())

    for key in ["boxes", "gt_boxes", "bboxes", "gt_bboxes"]:
        if key in sample:
            return len(sample[key])
        if key in targets:
            return len(targets[key])

    for key in ["heatmap", "hm"]:
        if key in targets:
            return int(targets[key].max().item() > 0)

    raise KeyError(
        f"Could not find object count. Sample keys: {list(sample.keys())}, "
        f"target keys: {list(targets.keys())}"
    )


def count_ds(ds):
    total = len(ds)
    with_person = 0
    total_persons = 0

    for i in range(total):
        n = num_persons(ds[i])
        if n > 0:
            with_person += 1
            total_persons += n

    pct = 100.0 * with_person / total if total > 0 else 0.0

    return {
        "total": total,
        "with_person": with_person,
        "pct": pct,
        "total_persons": total_persons,
    }


def print_row(split, direction, stats):
    print(
        f"{split.capitalize()} & {direction.capitalize()} & "
        f"{stats['total']} & {stats['with_person']} & "
        f"{stats['pct']:.1f}\\% & {stats['total_persons']} \\\\"
    )


@hydra.main(config_path="../configs", config_name="config_finetune", version_base=None)
def main(cfg: DictConfig):
    img_mean, img_std = load_norm(cfg)

    rows = []
    for split in ["train", "val"]:
        for direction in ["right", "left"]:
            ds = make_ds(cfg, direction, split, img_mean, img_std)
            stats = count_ds(ds)
            rows.append((split, direction, stats))

    print("\nDataset statistics:\n")
    for split, direction, stats in rows:
        print(
            f"{split:5s} {direction:5s} | "
            f"total={stats['total']:5d} | "
            f"with_person={stats['with_person']:5d} | "
            f"pct={stats['pct']:5.1f}% | "
            f"persons={stats['total_persons']:5d}"
        )

    print("\nLaTeX table:\n")
    print(r"\begin{table}[htbp]")
    print(r"    \centering")
    print(r"    \caption{Overview of the fine-tuning dataset.}")
    print(r"    \label{tab:finetuning_dataset}")
    print(r"    \begin{tabular}{llcccc}")
    print(r"        \toprule")
    print(r"        Split & View & Total samples & Samples with human & Human samples [\%] & Total humans \\")
    print(r"        \midrule")

    for split, direction, stats in rows:
        print("        ", end="")
        print_row(split, direction, stats)

    print(r"        \bottomrule")
    print(r"    \end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()