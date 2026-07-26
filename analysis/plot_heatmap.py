"""
Qualitative anomaly-heatmap composite for a single scan (cosine version).

Rebuilds the brain scan_0031 figure we had for full-covariance, but using the paper's
headline score (cosine_distance_map) instead of the log-likelihood map. Runs on the cluster
(needs the run's test_stats). Produces a 3-panel PNG:

    [ MRI slice ] [ cosine anomaly map ] [ overlay + ground-truth lesion contour ]

The scan is picked by its index in the split (the old composites were named scan_0031 =
index 31 of the anomalous test set; test order is deterministic, so the same index is the
same scan across runs). Examples (from repo root):

    $PY analysis/plot_heatmap.py --run "$RUNB" --index 31 --tag brain_diag_dpmm
    $PY analysis/plot_heatmap.py --run "$R_K100" --index 31 --tag brain_diag_K100

Output: analysis/out/heatmap_{tag}_scan_{index:04d}.png
"""

import os
import argparse
import numpy as np
import torch
from torch.nn import functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "analysis/out"


def parse_args():
    p = argparse.ArgumentParser(description="Single-scan cosine anomaly heatmap composite.")
    p.add_argument("--run", required=True, help="run folder with test_stats_{split}.pth")
    p.add_argument("--index", type=int, default=31, help="scan index within the split")
    p.add_argument("--split", default="anomalous", choices=["anomalous", "normal"],
                   help="which test split the scan is in (anomalous by default)")
    p.add_argument("--map", default="cosine_distance_map", help="score map to visualize")
    p.add_argument("--tag", default="", help="output filename suffix")
    return p.parse_args()


def to_gray(img):
    """(3,H,W) preprocessed tensor -> (H,W) display image via per-image min-max."""
    g = img.float().mean(0)
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    return g.numpy()


def main():
    args = parse_args()
    stats = torch.load(os.path.join(args.run, f"test_stats_{args.split}.pth"),
                       map_location="cpu", weights_only=False, mmap=True)

    i = args.index
    n = stats["image"].shape[0]
    if not (0 <= i < n):
        raise SystemExit(f"index {i} out of range (split has {n} images)")

    path = stats["image_paths"][i] if "image_paths" in stats else "?"
    print(f"scan index {i} of {n}  ({args.split})  ->  {path}")

    img = to_gray(stats["image"][i])                       # (H, W)
    H, W = img.shape

    amap = stats[args.map][i:i + 1].float()                # (1,1,32,32)
    amap = F.interpolate(amap, size=(H, W), mode="bilinear", align_corners=False)
    amap = amap[0, 0].numpy()                              # (H, W), higher = more anomalous

    # ground-truth lesion mask (resize to image resolution for the contour)
    gt = None
    if "labels" in stats:
        lab = stats["labels"][i:i + 1].float()             # (1,1,Hl,Wl)
        lab = F.interpolate(lab, size=(H, W), mode="nearest")[0, 0].numpy()
        if lab.max() > 0:
            gt = lab

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.8))
    for a in ax:
        a.axis("off")

    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("MRI slice", fontsize=13)

    im = ax[1].imshow(amap, cmap="magma")
    ax[1].set_title("cosine anomaly map", fontsize=13)
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)

    ax[2].imshow(img, cmap="gray")
    ax[2].imshow(amap, cmap="magma", alpha=0.5)
    if gt is not None:
        ax[2].contour(gt, levels=[0.5], colors="#39FF14", linewidths=1.5)
        ax[2].set_title("overlay + GT lesion (green)", fontsize=13)
    else:
        ax[2].set_title("overlay", fontsize=13)

    fig.suptitle(f"scan {i:04d} — {args.map}", fontsize=15, y=1.02)
    fig.tight_layout()

    os.makedirs(OUT, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = os.path.join(OUT, f"heatmap{suffix}_scan_{i:04d}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
