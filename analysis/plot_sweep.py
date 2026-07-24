"""
DPMM-vs-GMM sweep figure (contribution Step 2b).

Plots anomaly-detection performance of the fixed-K GMM as a function of K, with the
DPMM -- which auto-selects the number of components -- drawn as a horizontal reference
line (it has no K to choose). This is the figure that tests the paper's central
"no need to tune K" claim on RESC.

Inputs (produced by image_level.py / pixel_level.py, one file per model):
  pixel_level_results{tag}.csv      pixel-level AUROC/AUPR per score map
  image_level_scores{tag}.npz       pooled image scores + y_true
where {tag} is "" for the DPMM and "_gmm_K10", "_gmm_K50", "_gmm_K86", "_gmm_K150"
for the GMM sweep. Download these small files from the cluster into analysis/out/ first.

Output: analysis/out/fig_sweep_dpmm_vs_gmm.png

matplotlib + sklearn only, so it runs on the local Anaconda as-is.
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # render straight to file, no display needed
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

OUT = "analysis/out"

KS = [10, 50, 86, 150]        # the GMM sweep points (x-axis)
DPMM_K = 86                   # the DPMM's auto-selected component count (for annotation only)

# Which score to headline. anomaly_map = the mixture log-likelihood (the paper's main map).
SCORE_MAP = "anomaly_map"
IMAGE_POOL = "topk"           # best pooling for the shape-aware log-likelihood (from Step 2a)


def pixel_metrics(tag):
    """Read pixel-level (AUROC%, AUPR%) for SCORE_MAP from pixel_level_results{tag}.csv."""
    path = os.path.join(OUT, f"pixel_level_results{tag}.csv")
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["score_map"] == SCORE_MAP:
                return float(row["auroc_pct"]), float(row["aupr_pct"])
    raise KeyError(f"{SCORE_MAP} not found in {path}")


def image_metrics(tag):
    """Compute image-level (AUROC%, AUPR%) for SCORE_MAP+IMAGE_POOL from image_level_scores{tag}.npz."""
    data = np.load(os.path.join(OUT, f"image_level_scores{tag}.npz"))
    y_true = data["y_true"]
    y_score = data[f"{SCORE_MAP}__{IMAGE_POOL}"]
    return 100.0 * roc_auc_score(y_true, y_score), 100.0 * average_precision_score(y_true, y_score)


def set_ylim(ax, values, pad=1.5, min_span=8.0):
    """Zoom to the data but keep at least `min_span` percentage points on the axis, so tiny
    differences are not visually exaggerated into looking like a big effect (honest scaling)."""
    lo, hi = min(values), max(values)
    if hi - lo < min_span:                 # data is nearly flat -> enforce a minimum window
        mid = 0.5 * (lo + hi)
        lo, hi = mid - 0.5 * min_span, mid + 0.5 * min_span
    ax.set_ylim(lo - pad, hi + pad)


def panel(ax, gmm_vals, dpmm_val, title, ylabel):
    """One subplot: GMM value vs K (line + markers) with the DPMM as a horizontal reference."""
    ax.plot(KS, gmm_vals, "o-", color="tab:blue", label="GMM (fixed K)")
    ax.axhline(dpmm_val, color="tab:green", ls="--", lw=1.5,
               label=f"DPMM (auto K≈{DPMM_K}) = {dpmm_val:.2f}")
    ax.axvline(DPMM_K, color="grey", ls=":", lw=1, alpha=0.7)   # mark the DPMM's K
    for k, v in zip(KS, gmm_vals):
        ax.annotate(f"{v:.1f}", (k, v), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=7)
    ax.set_xticks(KS)
    ax.set_xlabel("number of GMM components K")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    set_ylim(ax, list(gmm_vals) + [dpmm_val])
    ax.legend(fontsize=8, loc="lower right")


def main():
    # DPMM = the untagged files; GMM = one tagged file per K.
    dpmm_px_auroc, dpmm_px_aupr = pixel_metrics("")
    dpmm_im_auroc, dpmm_im_aupr = image_metrics("")

    gmm_px = {k: pixel_metrics(f"_gmm_K{k}") for k in KS}
    gmm_im = {k: image_metrics(f"_gmm_K{k}") for k in KS}

    # --- console summary table ---
    print(f"score = {SCORE_MAP}  (image pooling = {IMAGE_POOL})\n")
    print(f"{'model':>10} | {'px AUROC':>8} {'px AUPR':>8} | {'im AUROC':>8} {'im AUPR':>8}")
    print("-" * 54)
    print(f"{'DPMM':>10} | {dpmm_px_auroc:8.2f} {dpmm_px_aupr:8.2f} | "
          f"{dpmm_im_auroc:8.2f} {dpmm_im_aupr:8.2f}")
    for k in KS:
        print(f"{'GMM K=' + str(k):>10} | {gmm_px[k][0]:8.2f} {gmm_px[k][1]:8.2f} | "
              f"{gmm_im[k][0]:8.2f} {gmm_im[k][1]:8.2f}")

    # --- 2x2 figure: rows = {AUROC, AUPR}, cols = {pixel, image} ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panel(axes[0, 0], [gmm_px[k][0] for k in KS], dpmm_px_auroc,
          "Pixel-level (localization)", "AUROC [%]")
    panel(axes[0, 1], [gmm_im[k][0] for k in KS], dpmm_im_auroc,
          "Image-level (triage)", "AUROC [%]")
    panel(axes[1, 0], [gmm_px[k][1] for k in KS], dpmm_px_aupr,
          "Pixel-level (localization)", "AUPR [%]")
    panel(axes[1, 1], [gmm_im[k][1] for k in KS], dpmm_im_aupr,
          "Image-level (triage)", "AUPR [%]")

    fig.suptitle("RESC: GMM performance vs K, with the DPMM (auto-K) as reference "
                 f"— score = {SCORE_MAP}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(OUT, "fig_sweep_dpmm_vs_gmm.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
