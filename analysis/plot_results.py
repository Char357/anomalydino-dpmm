"""
Make presentation figures from the image-level and pixel-level results.

Inputs (produced by the other two analysis scripts):
  analysis/out/image_level_scores.npz   -> pooled image scores + y_true (image_level.py)
  analysis/out/pixel_level_results.csv  -> per-map pixel AUROC/AUPR   (pixel_level.py)

Outputs (saved into analysis/out/):
  fig_pooling_ablation.png   grouped bar: image-level AUROC for every map x pooling
  fig_roc_curves.png         image-level ROC curves for the best map, 3 poolings
  fig_score_histogram.png    pooled-score separation, normal vs anomalous
  fig_pixel_vs_paper.png     pixel-level AUROC per map vs the paper's 90.20

matplotlib only (no seaborn), so it runs in the addpmm env as-is.

Author: Charlotte von Roznowski
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # no display on the cluster -> render straight to file
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

OUT = "analysis/out"

# Maps ordered shape-aware first, then shape-blind, with short labels for the axes.
MAP_ORDER = [
    ("anomaly_map", "log-lik"),
    ("max_log_prob_map", "max-log-lik"),
    ("weighted_distance_map", "Mahalanobis"),
    ("squared_weighted_distance_map", "Mahalanobis²"),
    ("distance_map", "L2"),
    ("squared_distance_map", "L2²"),
    ("cosine_distance_map", "cosine"),
]
METHODS = ["max", "mean", "topk"]
PAPER_RESC_AUROC = 90.20   # Table 1, "Ours", RESC pixel-level


def figure_pooling_ablation(scores, y_true):
    """Grouped bar chart of image-level AUROC: each map, one bar per pooling method.
    This is the core pooling-ablation figure (Step 2a): it shows top-k/max beating
    mean for the shape-aware scores, and the reversal for the raw distances."""
    labels = [short for _, short in MAP_ORDER]
    x = np.arange(len(MAP_ORDER))
    width = 0.26

    fig, ax = plt.subplots(figsize=(11, 5))
    for j, method in enumerate(METHODS):
        vals = [100.0 * roc_auc_score(y_true, scores[f"{mk}__{method}"])
                for mk, _ in MAP_ORDER]
        bars = ax.bar(x + (j - 1) * width, vals, width, label=method)
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)

    ax.axhline(50, color="grey", ls="--", lw=1, label="random (50)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Image-level AUROC [%]")
    ax.set_ylim(45, 95)
    ax.set_title("Image-level pooling ablation on RESC (patch → image score)")
    ax.legend(title="pooling", ncol=4, loc="lower center")
    ax.axvspan(-0.5, 3.5, color="tab:green", alpha=0.05)   # shade the shape-aware group
    ax.axvspan(3.5, 6.5, color="tab:red", alpha=0.05)      # shade the shape-blind group
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_pooling_ablation.png"), dpi=150)
    plt.close(fig)


def figure_roc_curves(scores, y_true, map_key="anomaly_map"):
    """Image-level ROC curves for the best score map, one line per pooling method.
    A classic view that makes the pooling effect visible as curve separation."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for method in METHODS:
        s = scores[f"{map_key}__{method}"]
        fpr, tpr, _ = roc_curve(y_true, s)
        auc = 100.0 * roc_auc_score(y_true, s)
        ax.plot(fpr, tpr, label=f"{method}  (AUROC {auc:.1f})")
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Image-level ROC on RESC — score = {map_key}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_roc_curves.png"), dpi=150)
    plt.close(fig)


def figure_score_histogram(scores, y_true, map_key="anomaly_map", method="topk"):
    """Distribution of the pooled image score for normal vs anomalous images.
    Shows *why* the AUROC is high: the two populations are well separated."""
    s = scores[f"{map_key}__{method}"]
    normal = s[y_true == 0]
    anomalous = s[y_true == 1]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(normal, bins=50, density=True, histtype="step", label="normal", color="tab:blue")
    ax.hist(anomalous, bins=50, density=True, histtype="step", label="anomalous", color="tab:orange")
    ax.set_xlabel(f"pooled image score ({method} of {map_key})")
    ax.set_ylabel("density")
    ax.set_title("Separation of image-level scores on RESC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_score_histogram.png"), dpi=150)
    plt.close(fig)


def figure_pixel_vs_paper():
    """Bar chart of our pixel-level AUROC per map, with a line at the paper's 90.20.
    Visual proof that our 20-epoch reproduction matches (exceeds) the paper."""
    rows = {}
    with open(os.path.join(OUT, "pixel_level_results.csv")) as f:
        for r in csv.DictReader(f):
            rows[r["score_map"]] = float(r["auroc_pct"])

    labels = [short for _, short in MAP_ORDER]
    vals = [rows[mk] for mk, _ in MAP_ORDER]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, vals, color="tab:purple")
    ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.axhline(PAPER_RESC_AUROC, color="black", ls="--", lw=1.2,
               label=f"paper (Table 1): {PAPER_RESC_AUROC:.2f}")
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Pixel-level AUROC [%]")
    ax.set_ylim(80, 95)
    ax.set_title("Pixel-level reproduction on RESC (20 epochs) vs paper (40 epochs)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_pixel_vs_paper.png"), dpi=150)
    plt.close(fig)


def main():
    data = np.load(os.path.join(OUT, "image_level_scores.npz"))
    y_true = data["y_true"]
    scores = {k: data[k] for k in data.files if k != "y_true"}

    figure_pooling_ablation(scores, y_true)
    figure_roc_curves(scores, y_true)
    figure_score_histogram(scores, y_true)
    figure_pixel_vs_paper()

    print("wrote 4 figures to", OUT)
    for name in ["fig_pooling_ablation.png", "fig_roc_curves.png",
                 "fig_score_histogram.png", "fig_pixel_vs_paper.png"]:
        print("  ", os.path.join(OUT, name))


if __name__ == "__main__":
    main()
