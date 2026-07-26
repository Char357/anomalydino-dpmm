"""
Qualitative anomaly-heatmap composite for a single scan (cosine version, dark theme).

Rebuilds the brain scan composite we had for full-covariance, but using the paper's headline
score (cosine_distance_map) instead of log-likelihood. Runs on the cluster (needs the run's
test_stats). 3-panel figure matching the old style:

    [ input scan ] [ ground truth (lesion) ] [ anomaly heatmap (cosine) ]

Pick the scan either by index, or with --top-lesion to auto-select the anomalous scan with the
LARGEST ground-truth lesion (a clear, illustrative example — recommended, since the old
"scan_0031" numbering was a curated selection, not the test index). Examples (repo root):

    $PY analysis/plot_heatmap.py --run "$RUNB" --top-lesion --tag brain_diag_dpmm
    $PY analysis/plot_heatmap.py --run "$RUNB" --index 31   --tag brain_diag_dpmm

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

# dark theme to match the old composites
BG = "#17131f"
FG = "#f2eefc"


def parse_args():
    p = argparse.ArgumentParser(description="Single-scan cosine anomaly heatmap composite (dark).")
    p.add_argument("--run", required=True, help="run folder with test_stats_{split}.pth")
    p.add_argument("--index", type=int, default=31, help="scan index within the split")
    p.add_argument("--top-lesion", action="store_true",
                   help="ignore --index; pick the anomalous scan with the largest GT lesion")
    p.add_argument("--split", default="anomalous", choices=["anomalous", "normal"])
    p.add_argument("--map", default="cosine_distance_map", help="score map to visualize")
    p.add_argument("--name", default="BraTS", help="dataset name in the title")
    p.add_argument("--tag", default="", help="output filename suffix")
    return p.parse_args()


def to_gray(img):
    g = img.float().mean(0)
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    return g.numpy()


def main():
    args = parse_args()
    stats = torch.load(os.path.join(args.run, f"test_stats_{args.split}.pth"),
                       map_location="cpu", weights_only=False, mmap=True)

    labels = stats["labels"]                                  # (N,1,Hl,Wl)
    n = stats["image"].shape[0]

    if args.top_lesion:
        # fraction of positive pixels per image; pick the biggest lesion
        frac = (labels.reshape(n, -1) > 0.5).float().mean(1)
        i = int(torch.argmax(frac))
        print(f"--top-lesion -> index {i} (lesion covers {100*frac[i]:.1f}% of the frame)")
    else:
        i = args.index
    if not (0 <= i < n):
        raise SystemExit(f"index {i} out of range (split has {n} images)")

    path = stats["image_paths"][i] if "image_paths" in stats else "?"
    print(f"scan index {i} of {n}  ({args.split})  ->  {path}")

    img = to_gray(stats["image"][i])
    H, W = img.shape

    amap = stats[args.map][i:i + 1].float()
    amap = F.interpolate(amap, size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy()

    lab = F.interpolate(labels[i:i + 1].float(), size=(H, W), mode="nearest")[0, 0].numpy()
    gt = np.ma.masked_where(lab < 0.5, lab)                   # show lesion only

    plt.rcParams.update({"text.color": FG, "axes.titlecolor": FG})
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 5.0), facecolor=BG)
    for a in ax:
        a.axis("off")
        a.set_facecolor(BG)

    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("input scan", fontsize=14)

    ax[1].imshow(img, cmap="gray")
    ax[1].imshow(gt, cmap="spring", alpha=0.9, vmin=0, vmax=1)   # lesion in bright magenta/green
    ax[1].set_title("ground truth (lesion)", fontsize=14)

    im = ax[2].imshow(amap, cmap="magma")
    ax[2].contour(lab, levels=[0.5], colors="#39FF14", linewidths=1.2)
    ax[2].set_title("anomaly heatmap (cosine)", fontsize=14)
    cb = fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color=FG)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=FG)

    fig.suptitle(f"{args.name} scan {i:04d} — cosine distance", fontsize=17,
                 fontweight="bold", color=FG, y=1.0)
    fig.tight_layout()

    os.makedirs(OUT, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = os.path.join(OUT, f"heatmap{suffix}_scan_{i:04d}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
