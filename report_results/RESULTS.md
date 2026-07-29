# Results — code extensions (diagonal-covariance reproduction of AnomalyDINO-DPMM)

All numbers are single-seed (seed 0), computed with the streaming 5%-subsample pixel-metric
estimator (`analysis/pixel_level.py`) and the image-level pooling script (`analysis/image_level.py`).
Score = **cosine distance to nearest cluster** (the paper's headline score) unless noted.
Figures are in `figures/`, qualitative heatmaps in `heatmaps/`.

## 1. Pixel-level reproduction — DPMM vs. the paper (cosine)

| Dataset | Metric | Paper | Ours (diag DPMM) | Δ |
|---|---|---:|---:|---:|
| RESC (retina OCT) | AUROC | 90.20 | 89.81 | −0.39 |
| RESC | AUPR | 41.66 | 41.59 | −0.07 |
| BraTS2021 (brain MRI) | AUROC | 96.21 | 95.23 | −0.98 |
| BraTS2021 | AUPR | 43.43 | 15.54* | −27.89* |

\*Brain AUPR is base-rate-dependent and **not comparable**: our metric scores all pixels including the
large background (~1.1% anomalous), which deflates AUPR. AUROC (base-rate-independent) reproduces to
within ~1 pp. Our DPMM selects ~102 (RESC) / ~107 (BraTS) clusters vs. the paper's 120–150, which
explains the small AUROC gap (see the K-sweep).

## 2. Score-function ablation (diag DPMM, pixel-level AUROC)

| Dataset | Likelihood | Euclidean | **Cosine** | (paper cosine) |
|---|---:|---:|---:|---:|
| RESC | 89.29 | 89.64 | **89.81** | 90.20 |
| BraTS2021 | 94.64 | 95.20 | **95.23** | 96.21 |

Cosine is best on both, matching the paper's ordering.

## 3. DPMM vs. fixed-K GMM sweep (pixel-level cosine AUROC)

| Model | RESC AUROC | RESC AUPR | BraTS AUROC | BraTS AUPR |
|---|---:|---:|---:|---:|
| GMM K=10 | 80.80 | 29.95 | 91.78 | 8.94 |
| GMM K=100 | 90.20 | 42.03 | 94.76 | 13.86 |
| GMM K=150 | 91.98 | 44.87 | 95.17 | 15.71 |
| DPMM (auto K≈102/107) | 89.81 | 41.59 | 95.23 | 15.54 |

K matters (K=10 collapses). Fixed K≈100 matches the DPMM. The DPMM's auto-K is near-optimal: at the
plateau for BraTS (~107), a slight undershoot for RESC (still rising at K=150).
Figures: `figures/fig_sweep_dpmm_vs_gmm_diag.png` (RESC), `figures/fig_sweep_dpmm_vs_gmm_brain_diag.png` (brain).

## 4. Image-level pooling ablation (cosine map, AUROC)

| Model | RESC max | RESC mean | RESC top-k | BraTS max | BraTS mean | BraTS top-k |
|---|---:|---:|---:|---:|---:|---:|
| GMM K=10 | 60.18 | 81.55 | 66.48 | 71.14 | 71.03 | 70.14 |
| GMM K=100 | 81.36 | 80.81 | 85.60 | 69.34 | 71.52 | 76.64 |
| GMM K=150 | 81.18 | 80.85 | 85.71 | 72.97 | 71.88 | 79.51 |
| DPMM | 81.95 | 79.84 | 85.49 | 72.46 | 71.33 | 80.50 |

Top-k is the best pooling for well-fit models (K≥100 and the DPMM). Best overall (any map): RESC
`anomaly_map`+top-k = 86.76; BraTS `cosine`+top-k = 80.50. Image-level (triage) and pixel-level
(localisation) are different tasks and their AUROCs are not directly comparable.

## 5. Runtime & memory (diagonal, 40 epochs, NVIDIA A100)

| Model | Wall time | Host RAM | Peak GPU |
|---|---|---:|---:|
| DPMM (500-trunc), RESC | 4:32 | 41.5 GB | 28.9 GB |
| DPMM (500-trunc), BraTS | 4:31 | 48.8 GB | 29.0 GB |
| GMM K=100, RESC | 3:47 | 39.8 GB | 13.8 GB |
| GMM K=100, BraTS | 3:20 | 46.2 GB | 13.6 GB |
| GMM K=10, RESC | 3:32 | 35.1 GB | 8.7 GB |

The DPMM's 500-component truncation costs ~1 h and ~2× the GPU memory of a K=100 GMM that scores the
same — the DP machinery has a real compute cost.
