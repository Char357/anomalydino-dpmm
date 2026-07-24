"""
Image-level (sample-wise) anomaly detection evaluation for AnomalyDINO-DPMM.

The released repository only computes PIXEL-level metrics (src/test.py upsamples
each anomaly map to the ground-truth mask resolution and scores every pixel).
The paper mentions image-level AUROC/AUPR only as a hidden step inside a
significance test and never tabulates or releases code for it.

This script fills that gap and loads the anomaly maps saved by a finished run
and for each per-patch score map POOLS the 32x32 grid into a single score per
image (max / mean / top-k), then computes image-level AUROC and AUPR against the
binary label (normal = 0, anomalous = 1).

Comparing the pooling methods is the pooling ablation (contribution, Step 2a).

"""

import os
import argparse
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

# Default run folder (the DPMM), containing test_stats_{normal,anomalous}.pth.
# Override on the command line with --run to point at a GMM run instead.
DEFAULT_RUN = ("results/AnomalyDINODPMM/RESC/dinov2_vits14_dpmm_448_500_full_pca_-1/"
               "-1-shot_preprocess=agnostic_objectmask=False_normalize_True_pos_enc_False/"
               "seed_0_")

# Single-channel per-patch score maps. For all of these, HIGHER = MORE ANOMALOUS.
MAP_KEYS = [
    "anomaly_map",                  # negative log-likelihood under the DPMM (main score)
    "distance_map",                 # distance to nearest cluster
    "cosine_distance_map",          # cosine distance to nearest cluster
    "weighted_distance_map",        # covariance-weighted (Mahalanobis-like) distance
    "max_log_prob_map",             # neg log-prob of the best-fitting component
    "squared_distance_map",
    "squared_weighted_distance_map",
]

METHODS = ["max", "mean", "topk"]   # topk = mean of the top TOPK_FRAC of patches
TOPK_FRAC = 0.01                    # top 1% of foreground patches (~10 of 1024)


def parse_args():
    p = argparse.ArgumentParser(
        description="Image-level (sample-wise) evaluation from saved test stats.")
    p.add_argument("--run", default=DEFAULT_RUN,
                   help="run folder containing test_stats_{normal,anomalous}.pth")
    p.add_argument("--tag", default="",
                   help="label appended to the output filename (e.g. gmm_K86); "
                        "empty keeps the DPMM default image_level_scores.npz")
    return p.parse_args()


def load(run, name):
    """Load a stats file lazily (mmap) so the big image/label tensors stay on disk."""
    return torch.load(os.path.join(run, name), map_location="cpu",
                      weights_only=False, mmap=True)


def pool(maps, masks, method):
    """Pool each (1, 32, 32) score map to one scalar per image.

    maps:  (N, 1, 32, 32) float score map
    masks: (N, 1, 32, 32) bool  foreground (object) mask
    Only foreground patches are pooled, so background does not dilute the score.
    """
    N = maps.shape[0]
    maps = maps.reshape(N, -1).float()
    masks = masks.reshape(N, -1).bool()
    out = torch.empty(N)
    for i in range(N):
        vals = maps[i][masks[i]]              # keep only foreground patches
        if vals.numel() == 0:                 # safety: image with no foreground
            vals = maps[i]
        if method == "max":
            out[i] = vals.max()               # single most-anomalous patch
        elif method == "mean":
            out[i] = vals.mean()              # average anomaly over the object
        elif method == "topk":
            k = max(1, int(round(TOPK_FRAC * vals.numel())))
            out[i] = vals.topk(k).values.mean()   # average of the k most-anomalous
        else:
            raise ValueError(method)
    return out


def main():
    args = parse_args()
    print(f"run: {args.run}")
    normal = load(args.run, "test_stats_normal.pth")
    anom = load(args.run, "test_stats_anomalous.pth")

    n_normal = normal["num_samples"]
    n_anom = anom["num_samples"]
    print(f"test images: {n_normal} normal + {n_anom} anomalous = {n_normal + n_anom}")

    # How much of each image is foreground? (sanity check on the object mask)
    fg = torch.cat([normal["object_mask"].reshape(n_normal, -1).float().mean(1),
                    anom["object_mask"].reshape(n_anom, -1).float().mean(1)])
    print(f"mean foreground fraction: {fg.mean():.3f}\n")

    # Image-level ground truth: normal = 0, anomalous = 1
    y_true = torch.cat([torch.zeros(n_normal), torch.ones(n_anom)]).numpy()

    rows = []          # (auroc, aupr, map_key, method) for the leaderboard
    scores = {}        # pooled scores, saved for plotting later

    print(f"{'score_map':30s} {'pool':6s} {'AUROC':>7s} {'AUPR':>7s}")
    print("-" * 54)
    for mk in MAP_KEYS:
        for method in METHODS:
            s_n = pool(normal[mk], normal["object_mask"], method)
            s_a = pool(anom[mk], anom["object_mask"], method)
            y_score = torch.cat([s_n, s_a]).numpy()
            auroc = roc_auc_score(y_true, y_score)
            aupr = average_precision_score(y_true, y_score)
            rows.append((auroc, aupr, mk, method))
            scores[f"{mk}__{method}"] = y_score
            print(f"{mk:30s} {method:6s} {auroc:7.4f} {aupr:7.4f}")

    # Leaderboard: which (score map, pooling) combination wins?
    print("\n=== ranked by AUROC (best first) ===")
    print(f"{'rank':4s} {'score_map':30s} {'pool':6s} {'AUROC':>7s} {'AUPR':>7s}")
    print("-" * 60)
    for rank, (auroc, aupr, mk, method) in enumerate(sorted(rows, reverse=True), 1):
        print(f"{rank:<4d} {mk:30s} {method:6s} {auroc:7.4f} {aupr:7.4f}")

    print(f"\nrandom-baseline: AUROC = 0.5000, AUPR = {y_true.mean():.4f}")

    os.makedirs("analysis/out", exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = f"analysis/out/image_level_scores{suffix}.npz"
    np.savez(out_path, y_true=y_true, **scores)
    print(f"saved pooled scores -> {out_path}")


if __name__ == "__main__":
    main()
