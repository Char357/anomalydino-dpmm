"""Makes a readable csv file for image-level results
Reads report_results/data/image_level_scores_*.npz, writes report_results/data/image_level_auroc.csv.
"""
import os
import glob
import csv
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

rows = []
for f in sorted(glob.glob(os.path.join(DATA, "image_level_scores_*.npz"))):
    run = os.path.basename(f)[len("image_level_scores_"):-len(".npz")]
    try:
        d = np.load(f, allow_pickle=True)
    except Exception as e:                 # e.g. the 0-byte diag_repro file (corrupted over-quota)
        print(f"SKIP {os.path.basename(f)} (corrupt/empty: {e})")
        continue
    y = d["y_true"]
    for key in d.files:
        if key == "y_true":
            continue
        s = d[key]
        score_map, pool = key.split("__")
        rows.append((run, score_map, pool,
                     100.0 * roc_auc_score(y, s),
                     100.0 * average_precision_score(y, s)))

out = os.path.join(DATA, "image_level_auroc.csv")
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["run", "score_map", "pool", "auroc_pct", "aupr_pct"])
    for run, mp, pool, auroc, aupr in rows:
        w.writerow([run, mp, pool, f"{auroc:.4f}", f"{aupr:.4f}"])

print(f"wrote {len(rows)} rows -> {out}")
