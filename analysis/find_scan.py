"""
Find the "ball-shaped tumor, upper-right hemisphere" scan (the old full-cov example) and
render COSINE heatmaps for the top candidates so you can pick the exact one. Runs on the
cluster (needs the run's test_stats_anomalous.pth).

Ranks anomalous scans by: lesion centroid in the upper-right, compact (round) shape, medium
size -- then writes a cosine heatmap composite for the top --n and prints their stats.

    $PY analysis/find_scan.py --run "$RUNB" --n 10 --tag ballfind
Then, if you want just the winner on its own: plot_heatmap.py --index <that idx>.
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
BG, FG = "#17131f", "#f2eefc"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--map", default="cosine_distance_map")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--name", default="BraTS")
    p.add_argument("--tag", default="ballfind")
    return p.parse_args()


def to_gray(img):
    g = img.float().mean(0)
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    return g.numpy()


def composite(stats, i, args, label="cosine"):
    img = to_gray(stats["image"][i])
    H, W = img.shape
    amap = F.interpolate(stats[args.map][i:i + 1].float(), size=(H, W),
                         mode="bilinear", align_corners=False)[0, 0].numpy()
    lab = F.interpolate(stats["labels"][i:i + 1].float(), size=(H, W), mode="nearest")[0, 0].numpy()
    gt = np.ma.masked_where(lab < 0.5, lab)

    plt.rcParams.update({"text.color": FG, "axes.titlecolor": FG})
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 5.0), facecolor=BG)
    for a in ax:
        a.axis("off")
        a.set_facecolor(BG)
    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("input scan", fontsize=14)
    ax[1].imshow(img, cmap="gray")
    ax[1].imshow(gt, cmap="spring", alpha=0.9, vmin=0, vmax=1)
    ax[1].set_title("ground truth (lesion)", fontsize=14)
    im = ax[2].imshow(amap, cmap="magma")
    ax[2].contour(lab, levels=[0.5], colors="#39FF14", linewidths=1.2)
    ax[2].set_title(f"anomaly heatmap ({label})", fontsize=14)
    cb = fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color=FG)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=FG)
    fig.suptitle(f"{args.name} scan {i:04d} — {label}", fontsize=17,
                 fontweight="bold", color=FG, y=1.0)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"heatmap_{args.tag}_scan_{i:04d}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out


def main():
    args = parse_args()
    s = torch.load(os.path.join(args.run, "test_stats_anomalous.pth"),
                   map_location="cpu", weights_only=False, mmap=True)
    labels = s["labels"]
    n = labels.shape[0]
    Hl, Wl = labels.shape[-2:]
    ys, xs = torch.meshgrid(torch.linspace(0, 1, Hl), torch.linspace(0, 1, Wl), indexing="ij")

    rows = []
    for i in range(n):
        m = labels[i, 0] > 0.5
        area = m.float().mean().item()
        if area < 0.01 or area > 0.12:            # skip tiny or whole-brain lesions
            continue
        cy = ys[m].mean().item()
        cx = xs[m].mean().item()
        yy, xx = torch.where(m)
        bb = float((yy.max() - yy.min() + 1) * (xx.max() - xx.min() + 1))
        fill = m.sum().item() / max(bb, 1.0)      # box fill = roundness/compactness proxy
        if 0.15 < cy < 0.5 and 0.5 < cx < 0.85:   # upper-right quadrant
            # target ~ (cy 0.33, cx 0.62), compact
            score = -abs(cy - 0.33) - abs(cx - 0.62) + 0.5 * fill
            rows.append((score, i, cy, cx, area, fill))

    rows.sort(reverse=True)
    print(f"{'rank':4s}{'idx':>7s}{'cy':>6s}{'cx':>6s}{'area%':>8s}{'fill':>7s}")
    made = []
    for r, (sc, i, cy, cx, area, fill) in enumerate(rows[:args.n], 1):
        print(f"{r:<4d}{i:>7d}{cy:>6.2f}{cx:>6.2f}{100 * area:>8.1f}{fill:>7.2f}")
        made.append(composite(s, i, args))
    print("\nsaved heatmaps:")
    for m in made:
        print("  " + m)


if __name__ == "__main__":
    main()
