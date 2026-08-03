"""
Pixel-level anomaly detection evaluation for AnomalyDINO-DPMM (Table 1 reproduction).

WHY THIS SCRIPT EXISTS:
The repository's own pixel-level metric loop (src/test.py -> calculate_metric)
does this:

    pred   = torch.cat([all normal maps, all anomalous maps])   # every image
    pred   = F.interpolate(pred, (1024, 512))                    # upsample ALL at once
    metric.update(pred[:, 0], labels[:, 0])                      # feed EVERY pixel

For RESC that is 1805 images x 1024 x 512 = ~9.5e8 pixels held in memory at once,
fed into torchmetrics 0.10.3, which additionally *buffers every pixel* to build the
ROC/PR curve. That needs ~65 GB of RAM, runs for hours, and crashed on AUPRO after 7+ hours.
(AUPRO is not even a Table 1 metric, so we drop it.)

HOW I WORKED AROUND THE BOTTLENECK:
Two independent ideas, both commented at the point they are used below:

  (1) STREAMING: I never hold all images at once. I processed ONE image at a time,
      upsample just that image, take what I need, and throw the rest away. Peak
      memory for the upsampling step is therefore one image (~0.5 M pixels), not
      one billion.

  (2) SUBSAMPLING: AUROC and AUPR are computed over a *population* of pixels. A
      uniform random subsample of that population is an UNBIASED estimator of the
      same AUROC/AUPR (it preserves the positive:negative ratio in expectation).
      Keeping e.g. 5% of pixels turns ~9.5e8 points into ~5e7, so a few hundred MB
      that scikit-learn sorts in seconds, while the estimate stays accurate to
      ~1e-3. This is the trick that replaces the 65 GB buffer.

I otherwise match the paper's protocol exactly: bilinear upsampling of each score
map to the ground-truth mask resolution, then score ALL pixels (background too,
just like the repo -> calculate_metric does not apply the object mask).

"""

import os
import argparse
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

# Default run folder (the DPMM). Override with --run to point at a GMM run instead.
DEFAULT_RUN = ("results/AnomalyDINODPMM/RESC/dinov2_vits14_dpmm_448_500_full_pca_-1/"
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Pixel-level (Table 1) evaluation from saved test stats.")
    p.add_argument("--run", default=DEFAULT_RUN,
                   help="run folder containing test_stats_{normal,anomalous}.pth")
    p.add_argument("--tag", default="",
                   help="label appended to the output filename (e.g. gmm_K86); "
                        "empty keeps the DPMM default pixel_level_results.csv")
    p.add_argument("--paper-auroc", type=float, default=90.20,
                   help="paper's pixel AUROC%% for this dataset (default = RESC 90.20; BraTS ~96)")
    p.add_argument("--paper-aupr", type=float, default=41.66,
                   help="paper's pixel AUPR%% for this dataset (default = RESC 41.66)")
    p.add_argument("--paper-name", default="RESC",
                   help="dataset name shown in the comparison line")
    return p.parse_args()


def load(run, name):
    """Load a stats file lazily (mmap): tensors stay on disk until we touch them,
    so opening the 8 GB / 6 GB files costs almost no RAM."""
    return torch.load(os.path.join(run, name), map_location="cpu",
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
    args = parse_args()
    print(f"run: {args.run}")
    torch.manual_seed(SEED)                  # reproducible subsample

    normal = load(args.run, "test_stats_normal.pth")
    anom = load(args.run, "test_stats_anomalous.pth")

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

    # The paper's headline anomaly score is the COSINE distance to the nearest
    # component mean (Eq. 11) -- NOT the likelihood, which the paper explicitly
    # rejects (Sec. 2.2). So the like-for-like reproduction number is the
    # cosine_distance_map row. We report that against the paper, and show the
    # best-scoring map only as a secondary note.
    by_map = {mk: (auroc, aupr) for auroc, aupr, mk in rows}
    cos_auroc, cos_aupr = by_map["cosine_distance_map"]
    best = max(rows)   # highest AUROC among all maps (secondary info only)
    print(f"\n=== reproduction vs paper ({args.paper_name}, Table 3 'Cosine/True' = the headline metric) ===")
    print(f"paper (cosine):  AUROC = {args.paper_auroc:.2f} %   AUPR = {args.paper_aupr:.2f} %")
    print(f"ours  (cosine):  AUROC = {cos_auroc:.2f} %   AUPR = {cos_aupr:.2f} %   <-- THE reproduction number")
    print(f"difference:      AUROC {cos_auroc - args.paper_auroc:+.2f} pp   "
          f"AUPR {cos_aupr - args.paper_aupr:+.2f} pp")
    print(f"\n(secondary) best-scoring map overall: {best[2]}  "
          f"AUROC {best[0]:.2f} %  AUPR {best[1]:.2f} %")

    os.makedirs("analysis/out", exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = f"analysis/out/pixel_level_results{suffix}.csv"
    with open(out_path, "w") as f:
        f.write("score_map,auroc_pct,aupr_pct\n")
        for auroc, aupr, mk in sorted(rows, reverse=True):
            f.write(f"{mk},{auroc:.4f},{aupr:.4f}\n")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
