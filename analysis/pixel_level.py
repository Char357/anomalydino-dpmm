"""
Pixel-level anomaly detection evaluation for AnomalyDINO-DPMM (Table 1 reproduction).

WHY THIS SCRIPT EXISTS
----------------------
The repository's own pixel-level metric loop (src/test.py -> calculate_metric)
does this:

    pred   = torch.cat([all normal maps, all anomalous maps])   # every image
    pred   = F.interpolate(pred, (1024, 512))                    # upsample ALL at once
    metric.update(pred[:, 0], labels[:, 0])                      # feed EVERY pixel

For RESC that is 1805 images x 1024 x 512 = ~9.5e8 pixels held in memory at once,
fed into torchmetrics 0.10.3, which additionally *buffers every pixel* to build the
ROC/PR curve. That needs ~65 GB of RAM, runs for hours, and crashed on AUPRO.
(AUPRO is not even a Table 1 metric, so we drop it.)

HOW WE WORK AROUND THE BOTTLENECK
---------------------------------
Two independent ideas, both commented at the point they are used below:

  (1) STREAMING: we never hold all images at once. We process ONE image at a time,
      upsample just that image, take what we need, and throw the rest away. Peak
      memory for the upsampling step is therefore one image (~0.5 M pixels), not
      one billion.

  (2) SUBSAMPLING: AUROC and AUPR are computed over a *population* of pixels. A
      uniform random subsample of that population is an UNBIASED estimator of the
      same AUROC/AUPR (it preserves the positive:negative ratio in expectation).
      Keeping e.g. 5% of pixels turns ~9.5e8 points into ~5e7 -- a few hundred MB
      that scikit-learn sorts in seconds -- while the estimate stays accurate to
      ~1e-3. This is the key trick that replaces the 65 GB buffer.

We otherwise match the paper's protocol exactly: bilinear upsampling of each score
map to the ground-truth mask resolution, then score ALL pixels (background too,
just like the repo -- calculate_metric does not apply the object mask).

Paper's target for RESC (Table 1, "Ours"):  AUROC = 90.20 %,  AUPR = 41.66 %
Our run used 20 epochs (the paper used 40); comparing to these numbers tells us
whether 20 epochs was enough before we build the rest of the analysis on it.

Author: Charlotte von Roznowski
"""

import os
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

RUN = ("results/AnomalyDINODPMM/RESC/dinov2_vits14_dpmm_448_500_full_pca_-1/"
       "-1-shot_preprocess=agnostic_objectmask=False_normalize_True_pos_enc_False/"
       "seed_0_")

# The same single-channel per-patch score maps as the image-level script.
# For all of these, HIGHER = MORE ANOMALOUS, so no sign-flipping is needed.
MAP_KEYS = [
    "anomaly_map", "distance_map", "cosine_distance_map",
    "weighted_distance_map", "max_log_prob_map",
    "squared_distance_map", "squared_weighted_distance_map",
]

SAMPLE_FRAC = 0.05      # keep 5% of pixels per image (the subsampling trick)
SEED = 0                # fixed seed -> the subsample (and thus the numbers) is reproducible

# Paper's RESC pixel-level numbers, for side-by-side comparison
PAPER_RESC = {"auroc": 90.20, "aupr": 41.66}


def load(name):
    """Load a stats file lazily (mmap): tensors stay on disk until we touch them,
    so opening the 8 GB / 6 GB files costs almost no RAM."""
    return torch.load(os.path.join(RUN, name), map_location="cpu",
                      weights_only=False, mmap=True)


def collect_pixels(stats):
    """Stream over one split (normal OR anomalous), one image at a time, and return
    a subsampled set of (per-map scores, label) pixel pairs.

    Returns:
        scores: dict map_key -> 1-D float tensor of sampled pixel scores
        labels: 1-D uint8 tensor of the matching ground-truth pixel labels (0/1)
    """
    labels_full = stats["labels"]            # (N, 1, 1024, 512) on disk (mmap)
    N = labels_full.shape[0]
    H, W = labels_full.shape[-2:]

    scores = {mk: [] for mk in MAP_KEYS}
    labels = []

    for i in range(N):
        # --- ground-truth mask for THIS image, flattened to a 1-D pixel vector ---
        lab = (labels_full[i, 0].reshape(-1) > 0.5).to(torch.uint8)   # (H*W,)

        # --- SUBSAMPLING: one random pixel mask, shared by the label and all 7 maps
        #     so every sampled score lines up with its own ground-truth pixel.
        keep = torch.rand(H * W) < SAMPLE_FRAC                          # (H*W,) bool
        labels.append(lab[keep])

        # --- STREAMING: upsample ONLY this image's maps, take the sampled pixels ---
        for mk in MAP_KEYS:
            m = stats[mk][i:i+1].float()                               # (1, 1, 32, 32)
            # bilinear upsample to mask resolution -- matches the repo's calculate_metric
            up = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)
            scores[mk].append(up.reshape(-1)[keep])                    # keep 5%, drop the rest

    scores = {mk: torch.cat(v) for mk, v in scores.items()}
    labels = torch.cat(labels)
    return scores, labels


def main():
    torch.manual_seed(SEED)                  # reproducible subsample

    normal = load("test_stats_normal.pth")
    anom = load("test_stats_anomalous.pth")

    print("collecting sampled pixels (streaming, 5% subsample)...")
    s_norm, y_norm = collect_pixels(normal)   # normal images: all labels are 0
    s_anom, y_anom = collect_pixels(anom)     # anomalous images: lesion pixels are 1

    # Combine both splits into one pixel population
    y_true = torch.cat([y_norm, y_anom]).numpy()
    n_pos = int(y_true.sum())
    n_tot = int(y_true.size)
    print(f"sampled {n_tot:,} pixels total, of which {n_pos:,} are anomalous "
          f"({100.0 * n_pos / n_tot:.3f} %)\n")

    print(f"{'score_map':30s} {'AUROC[%]':>9s} {'AUPR[%]':>9s}")
    print("-" * 50)
    rows = []
    for mk in MAP_KEYS:
        y_score = torch.cat([s_norm[mk], s_anom[mk]]).numpy()
        auroc = 100.0 * roc_auc_score(y_true, y_score)
        aupr = 100.0 * average_precision_score(y_true, y_score)
        rows.append((auroc, aupr, mk))
        print(f"{mk:30s} {auroc:9.2f} {aupr:9.2f}")

    # Which of our maps best matches the paper's headline number?
    best = max(rows)   # highest AUROC
    print("\n=== comparison to paper (RESC, Table 1 'Ours', 40 epochs) ===")
    print(f"paper:      AUROC = {PAPER_RESC['auroc']:.2f} %   AUPR = {PAPER_RESC['aupr']:.2f} %")
    print(f"ours(best): AUROC = {best[0]:.2f} %   AUPR = {best[1]:.2f} %   "
          f"[{best[2]}, 20 epochs]")
    print(f"difference: AUROC {best[0] - PAPER_RESC['auroc']:+.2f} pp   "
          f"AUPR {best[1] - PAPER_RESC['aupr']:+.2f} pp")

    os.makedirs("analysis/out", exist_ok=True)
    with open("analysis/out/pixel_level_results.csv", "w") as f:
        f.write("score_map,auroc_pct,aupr_pct\n")
        for auroc, aupr, mk in sorted(rows, reverse=True):
            f.write(f"{mk},{auroc:.4f},{aupr:.4f}\n")
    print("\nsaved -> analysis/out/pixel_level_results.csv")


if __name__ == "__main__":
    main()
