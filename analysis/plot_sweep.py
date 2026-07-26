"""
DPMM-vs-GMM sweep figure (contribution Step 2b).

Plots anomaly-detection performance of the fixed-K GMM as a function of K, with the
DPMM -- which auto-selects the number of components -- drawn as a horizontal reference
line (it has no K to choose). This is the figure that tests the paper's central
"no need to tune K" claim.

Inputs (produced by pixel_level.py / image_level.py, one file per model):
  pixel_level_results{tag}.csv      pixel-level AUROC/AUPR per score map
  image_level_scores{tag}.npz       pooled image scores + y_true
Download these small files into analysis/out/ first (or run this on the cluster where they live).

Example (diagonal-cov reproduction, RESC, cosine score):
  python analysis/plot_sweep.py --name "RESC (diag)" --ks 10 100 150 --dpmm-k 102 \
      --dpmm-tag _diag_repro --gmm-tag-prefix _RESC_diag_K --score cosine_distance_map \
      --out-suffix _diag

Output: analysis/out/fig_sweep_dpmm_vs_gmm{out-suffix}.png

matplotlib + sklearn only, so it runs on the local Anaconda as-is.
"""

import csv
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")   # render straight to file, no display needed
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

from plot_style import use_style, BLUE, GOLD, MUTED

OUT = "analysis/out"


def parse_args():
    p = argparse.ArgumentParser(description="DPMM-vs-GMM sweep figure (RESC or brain).")
    p.add_argument("--name", default="RESC",
                   help="dataset name shown in the title / output filename")
    p.add_argument("--ks", type=int, nargs="+", default=[10, 50, 86, 150],
                   help="GMM sweep points on the x-axis")
    p.add_argument("--dpmm-k", type=int, default=86,
                   help="the DPMM's auto-selected component count (annotation only)")
    p.add_argument("--dpmm-tag", default="",
                   help="filename tag of the DPMM results (e.g. _diag_repro / _brain_diag_repro)")
    p.add_argument("--gmm-tag-prefix", default="_gmm_K",
                   help="filename tag prefix of each GMM run; full tag = prefix+K "
                        "(diag: _RESC_diag_K / _bras2021_diag_K)")
    p.add_argument("--score", default="anomaly_map",
                   help="which score map to headline (diag reproduction: cosine_distance_map)")
    p.add_argument("--pool", default="topk", help="image-level pooling method to plot")
    p.add_argument("--title", default="", help="custom suptitle (else a neutral factual default)")
    p.add_argument("--out-suffix", default="",
                   help="suffix for the output PNG (e.g. _diag), so datasets don't overwrite")
    return p.parse_args()


def pixel_metrics(tag, score):
    """Read pixel-level (AUROC%, AUPR%) for `score` from pixel_level_results{tag}.csv."""
    path = os.path.join(OUT, f"pixel_level_results{tag}.csv")
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["score_map"] == score:
                return float(row["auroc_pct"]), float(row["aupr_pct"])
    raise KeyError(f"{score} not found in {path}")


def image_metrics(tag, score, pool):
    """Compute image-level (AUROC%, AUPR%) for score+pool from image_level_scores{tag}.npz."""
    data = np.load(os.path.join(OUT, f"image_level_scores{tag}.npz"))
    y_true = data["y_true"]
    y_score = data[f"{score}__{pool}"]
    return 100.0 * roc_auc_score(y_true, y_score), 100.0 * average_precision_score(y_true, y_score)


def set_ylim(ax, values, pad=1.5, min_span=8.0):
    """Zoom to the data but keep at least `min_span` percentage points on the axis, so tiny
    differences are not visually exaggerated into looking like a big effect (honest scaling)."""
    lo, hi = min(values), max(values)
    if hi - lo < min_span:                 # data is nearly flat -> enforce a minimum window
        mid = 0.5 * (lo + hi)
        lo, hi = mid - 0.5 * min_span, mid + 0.5 * min_span
    ax.set_ylim(lo - pad, hi + pad)


def panel(ax, ks, dpmm_k, gmm_vals, dpmm_val, title, ylabel):
    """One subplot: GMM value vs K (line + markers) with the DPMM as a horizontal reference."""
    ax.plot(ks, gmm_vals, "o-", color=BLUE, lw=2, ms=8, label="GMM (fixed K)")
    ax.axhline(dpmm_val, color=GOLD, ls="--", lw=2,
               label=f"DPMM (auto K≈{dpmm_k}) = {dpmm_val:.2f}")
    ax.axvline(dpmm_k, color=MUTED, ls=":", lw=1, alpha=0.6)   # mark the DPMM's K
    for k, v in zip(ks, gmm_vals):
        ax.annotate(f"{v:.1f}", (k, v), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
    ax.set_xticks(ks)
    ax.set_xlabel("number of GMM components K")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    set_ylim(ax, list(gmm_vals) + [dpmm_val])
    ax.legend(fontsize=8, loc="lower right")


def main():
    args = parse_args()
    ks = args.ks
    use_style()
    dpmm_px_auroc, dpmm_px_aupr = pixel_metrics(args.dpmm_tag, args.score)
    dpmm_im_auroc, dpmm_im_aupr = image_metrics(args.dpmm_tag, args.score, args.pool)

    gmm_px = {k: pixel_metrics(f"{args.gmm_tag_prefix}{k}", args.score) for k in ks}
    gmm_im = {k: image_metrics(f"{args.gmm_tag_prefix}{k}", args.score, args.pool) for k in ks}

    # --- console summary table ---
    print(f"dataset = {args.name}   score = {args.score}  (image pooling = {args.pool})\n")
    print(f"{'model':>10} | {'px AUROC':>8} {'px AUPR':>8} | {'im AUROC':>8} {'im AUPR':>8}")
    print("-" * 54)
    print(f"{'DPMM':>10} | {dpmm_px_auroc:8.2f} {dpmm_px_aupr:8.2f} | "
          f"{dpmm_im_auroc:8.2f} {dpmm_im_aupr:8.2f}")
    for k in ks:
        print(f"{'GMM K=' + str(k):>10} | {gmm_px[k][0]:8.2f} {gmm_px[k][1]:8.2f} | "
              f"{gmm_im[k][0]:8.2f} {gmm_im[k][1]:8.2f}")

    # --- 2x2 figure: rows = {AUROC, AUPR}, cols = {pixel, image} ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panel(axes[0, 0], ks, args.dpmm_k, [gmm_px[k][0] for k in ks], dpmm_px_auroc,
          "Pixel-level (localization)", "AUROC [%]")
    panel(axes[0, 1], ks, args.dpmm_k, [gmm_im[k][0] for k in ks], dpmm_im_auroc,
          "Image-level (triage)", "AUROC [%]")
    panel(axes[1, 0], ks, args.dpmm_k, [gmm_px[k][1] for k in ks], dpmm_px_aupr,
          "Pixel-level (localization)", "AUPR [%]")
    panel(axes[1, 1], ks, args.dpmm_k, [gmm_im[k][1] for k in ks], dpmm_im_aupr,
          "Image-level (triage)", "AUPR [%]")

    title = args.title or (f"Fixed-K GMM vs auto-K DPMM — {args.name}\n"
                           f"(score = {args.score}, image pooling = {args.pool})")
    fig.suptitle(title, fontsize=17)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(OUT, f"fig_sweep_dpmm_vs_gmm{args.out_suffix}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
