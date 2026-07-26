import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DIST = 1.0


def make_colors(names):
    base = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {name: base[i % len(base)] for i, name in enumerate(names)}


def iter_models(df, order):
    for name in order:
        g = df[df["model"] == name].copy()
        if not g.empty:
            if "score_thresh" in g.columns:
                g = g.sort_values("score_thresh")
            yield name, g


def check(files):
    for file in files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Missing file: {file}")


def save(fig, save_dir, name):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def load(files, names):
    data = []

    if names is None:
        names = [Path(f).stem for f in files]

    if len(files) != len(names):
        raise ValueError("Number of files and names must match")

    for file, name in zip(files, names):
        df = pd.read_csv(file)
        df["model"] = name
        data.append(df)

    return pd.concat(data, ignore_index=True)


def load_iou(files, names):
    data = []

    for file, name in zip(files, names):
        df = pd.read_csv(file)
        df["model"] = name
        data.append(df)

    return pd.concat(data, ignore_index=True)


def at_dist(df, dist=DIST):
    out = df[np.isclose(df["dist_thresh"], dist)].copy()
    if out.empty:
        raise ValueError(f"No rows found for dist_thresh={dist}")
    return out.sort_values(["model", "score_thresh"])


def plot_eval(df, save_dir, colors, order, dist=DIST):
    df = at_dist(df, dist)

    metrics = [
        ("precision", "Precision", "Metric value"),
        ("recall", "Recall", "Metric value"),
        ("f1", "F1", "Metric value"),
        ("fp_per_frame", "False positives / frame", "FP/frame"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, (key, title, ylabel) in zip(axes, metrics):
        for name, g in iter_models(df, order):
            ax.plot(
                g["score_thresh"],
                g[key],
                marker="o",
                label=name,
                color=colors[name],
            )

        ax.set_title(title)
        ax.set_xlabel("Score threshold")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        if key != "fp_per_frame":
            ax.set_ylim(0, 1)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(f"Evaluation curves, match distance = {dist:.1f} m", y=0.98)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 0.94),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.89])

    save(fig, save_dir, "evaluation_curves_compare.png")


def plot_ap_dist(df, save_dir, colors, order):
    fig, ax = plt.subplots(figsize=(7, 5))

    for name, g in iter_models(df, order):
        ap = (
            g.groupby("dist_thresh", as_index=False)["ap"]
            .first()
            .sort_values("dist_thresh")
        )

        ax.plot(
            ap["dist_thresh"],
            ap["ap"],
            marker="o",
            label=name,
            color=colors[name],
        )

    ax.set_xlabel("Center-distance threshold (m)")
    ax.set_ylabel("AP")
    ax.set_title("Center-distance AP")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    save(fig, save_dir, "ap_by_distance_compare.png")


def plot_err(df, save_dir, colors, order, dist=DIST):
    df = at_dist(df, dist)

    metrics = [
        ("mean_depth_error", "Forward depth error", "Error (m)"),
        ("mean_lateral_error", "Lateral error", "Error (m)"),
        ("mean_euclidean_error", "Euclidean center error", "Error (m)"),
        ("mean_bearing_error_deg", "Bearing error", "Error (deg)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, (key, title, ylabel) in zip(axes, metrics):
        for name, g in iter_models(df, order):
            ax.plot(
                g["score_thresh"],
                g[key],
                marker="o",
                label=name,
                color=colors[name],
            )

        ax.set_title(title)
        ax.set_xlabel("Score threshold")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(f"Localization errors, match distance = {dist:.1f} m", y=0.98)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 0.94),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.89])

    save(fig, save_dir, "localization_error_curves_compare.png")


def plot_best(df, save_dir, colors, order, dist=DIST):
    df = at_dist(df, dist)

    metrics = [
        ("mean_depth_error", "Depth"),
        ("mean_lateral_error", "Lateral"),
        ("mean_euclidean_error", "Euclidean"),
        ("mean_range_error", "Range"),
        ("mean_bearing_error_deg", "Bearing"),
    ]

    rows = []

    for name, g in iter_models(df, order):
        g = g.dropna(subset=["f1"])
        best = g.loc[g["f1"].idxmax()]

        for key, label in metrics:
            rows.append({
                "model": name,
                "metric": label,
                "value": best[key],
                "score": best["score_thresh"],
                "f1": best["f1"],
            })

    best_df = pd.DataFrame(rows)

    metric_names = [m[1] for m in metrics]
    model_names = [name for name in order if name in best_df["model"].unique()]

    x = np.arange(len(metric_names))
    width = 0.8 / max(len(model_names), 1)

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, name in enumerate(model_names):
        vals = [
            best_df[
                (best_df["model"] == name)
                & (best_df["metric"] == metric)
            ]["value"].iloc[0]
            for metric in metric_names
        ]

        ax.bar(
            x + i * width - width * (len(model_names) - 1) / 2,
            vals,
            width,
            label=name,
            color=colors[name],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Error (m or deg)")
    ax.set_title(f"Localization errors at best F1, match distance = {dist:.1f} m")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    save(fig, save_dir, "best_f1_error_summary_compare.png")


def plot_bias(df, save_dir, colors, order, dist=DIST):
    df = at_dist(df, dist)

    metrics = [
        ("mean_depth_bias", "Forward depth bias", "Bias (m)"),
        ("mean_lateral_bias", "Lateral bias", "Bias (m)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, (key, title, ylabel) in zip(axes, metrics):
        for name, g in iter_models(df, order):
            ax.plot(
                g["score_thresh"],
                g[key],
                marker="o",
                label=name,
                color=colors[name],
            )

        ax.axhline(0, linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Score threshold")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(f"Localization bias, match distance = {dist:.1f} m", y=0.98)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 0.91),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.84])

    save(fig, save_dir, "bias_curves_compare.png")


def plot_iou_ap(df, save_dir, colors, order):
    df = df.copy()
    df["iou_thresh"] = df["iou_thresh"].astype(float)

    ious = sorted(df["iou_thresh"].unique())
    models = [name for name in order if name in df["model"].unique()]

    x = np.arange(len(ious))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 5))

    max_val = 0

    for i, name in enumerate(models):
        g = df[df["model"] == name].sort_values("iou_thresh")
        vals = [g[g["iou_thresh"] == iou]["ap"].iloc[0] for iou in ious]
        max_val = max(max_val, max(vals))

        xpos = x + i * width - width * (len(models) - 1) / 2

        bars = ax.bar(
            xpos,
            vals,
            width,
            label=name,
            color=colors[name],
        )

        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{val * 100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"IoU {iou * 100:.0f}%" for iou in ious])
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("AP")
    ax.set_title("3D IoU-based AP comparison")
    ax.set_ylim(0, min(1.08, max_val + 0.12))
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    save(fig, save_dir, "ap_by_iou_compare.png")


def main():
    use_third = False

    models = [
        {
            "name": "2-modal PRETRAINED",
            "summary": "reports/figures/evaluation/evaluation_summary_rad_feat_pretrained.csv",
            "iou": "reports/figures/evaluation/iou_rad_feat_pretrained.csv",
        },
        {
            "name": "all-modal PRETRAINED",
            "summary": "reports/figures/evaluation/evaluation_summary_cam_rad_feat_pretrained.csv",
            "iou": "reports/figures/evaluation/iou_cam_rad_feat_pretrained.csv",
        },
        # {
        #     "name": "NOT PRRETRAINED",
        #     "summary": "reports/figures/evaluation/evaluation_summary_cam_rad_feat_NOTpretrained.csv",
        #     "iou": "reports/figures/evaluation/iou_cam_rad_feat_NOTpretrained.csv",
        # },
        # {
        #     "name": "NOT PRRETRAINED",
        #     "summary": "reports/figures/evaluation/evaluation_summary_rad_feat_NOTpretrained.csv",
        #     "iou": "reports/figures/evaluation/iou_rad_feat_NOTpretrained.csv",
        # },
    ]

    if use_third:
        models.append(
            {
                "name": "Radar + camera BEV + BEV features, no pretraining",
                "summary": "reports/figures/evaluation/evaluation_summary_cam_rad_feat_NOTpretrained.csv",
                "iou": "reports/figures/evaluation/iou_cam_rad_feat_NOTpretrained.csv",
            }
        )

    save_dir = "reports/figures/evaluation/evaluation_compare"
    os.makedirs(save_dir, exist_ok=True)

    files = [m["summary"] for m in models]
    names = [m["name"] for m in models]

    iou_files = [m["iou"] for m in models]
    iou_names = [m["name"] for m in models]

    colors = make_colors(names)
    order = names

    check(files)
    check(iou_files)

    df = load(files, names)
    iou_df = load_iou(iou_files, iou_names)

    plot_eval(df, save_dir, colors, order)
    plot_ap_dist(df, save_dir, colors, order)
    plot_err(df, save_dir, colors, order)
    plot_best(df, save_dir, colors, order)
    plot_bias(df, save_dir, colors, order)
    plot_iou_ap(iou_df, save_dir, colors, order)

    print(f"Saved plots to {save_dir}")


if __name__ == "__main__":
    main()