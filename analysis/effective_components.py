"""
Effective-number-of-components diagnostic — explains WHY the DPMM-vs-GMM sweep is flat.

Each fitted model stores its parameters in checkpoint_best.pth under the key "dpmm"
(train.py: save_checkpoint(..., dpmm=model.state_dict())). We reconstruct each model's
mixing weights pi and measure how many components actually carry weight.

Hypothesis: regardless of the *nominal* K, every fit ends up using only ~10-20 components
with non-negligible weight. If true, that's the mechanism behind the flat sweep — K=10 is
already enough, and the DPMM's 86 / the GMM's 150 are mostly near-empty.

Measures reported per model:
  active@1e-2 / 1e-3 : number of components with mixing weight pi above that threshold.
  perplexity         : exp(entropy(pi)) = a threshold-free "effective number of components".
                       (m equal components -> perplexity m; one dominant one -> perplexity 1.)

Runs on the CLUSTER (needs torch, src/, and the checkpoints). Run from the repo root:
    python analysis/effective_components.py                          # RESC (default)
    python analysis/effective_components.py --dataset bras2021 \
           --ks 10 86 150 --tag brain                                # brain (K=50 never run)
Prints a table, saves the pi spectra (for local plotting) and a spectrum figure.
"""

import os
import sys
import argparse

# Allow "import src..." no matter what the current working directory is.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_style import use_style, BLUE, PURPLE_RAMP, MUTED
from src.DirichletProcessMixture.dpmm import DPMM
from src.DirichletProcessMixture.finite_gmm import FiniteGMM

EXP = "-1-shot_preprocess=agnostic_objectmask=False_normalize_True_pos_enc_False"


def parse_args():
    p = argparse.ArgumentParser(
        description="Effective-number-of-components diagnostic (RESC or brain).")
    p.add_argument("--dataset", default="RESC",
                   help="folder under results/AnomalyDINODPMM/ (RESC or bras2021)")
    p.add_argument("--ks", type=int, nargs="+", default=[10, 50, 86, 150],
                   help="GMM K slots to include (brain never ran K=50 -> pass: 10 86 150)")
    p.add_argument("--dpmm-slot", type=int, default=500,
                   help="architecture K-slot of the DPMM run (always 500)")
    p.add_argument("--tag", default="",
                   help="suffix for output files (e.g. brain), so datasets don't overwrite")
    return p.parse_args()


def build_runs(ks, dpmm_slot):
    # (label, K-slot in the architecture name, seed folder, model class to reconstruct pi correctly)
    # The DPMM's pi comes from stick-breaking (v); the GMM's from normalised responsibilities.
    runs = [("DPMM", dpmm_slot, "seed_0_", DPMM)]
    for k in ks:
        runs.append((f"GMM K={k}", k, "seed_0_gmm", FiniteGMM))
    return runs


def run_dir(dataset, kslot, seed_folder):
    root = f"results/AnomalyDINODPMM/{dataset}"
    arch = f"dinov2_vits14_dpmm_448_{kslot}_full_pca_-1"
    return os.path.join(root, arch, EXP, seed_folder)


def load_pi(dataset, kslot, seed_folder, ModelClass):
    """Rebuild the model from its checkpoint and return its mixing weights, sorted descending."""
    path = os.path.join(run_dir(dataset, kslot, seed_folder), "checkpoint_best.pth")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["dpmm"]                             # state_dict, same key for DPMM and FiniteGMM
    K, D = sd["mean"].shape
    model = ModelClass(K=K, D=D, device="cpu")
    model.load_state_dict(sd)                     # uses the model's own calculate_pi below
    pi = model.calculate_pi().detach().cpu().numpy()
    return np.sort(pi)[::-1]                       # largest first -> a clean spectrum


def perplexity(pi):
    pi = pi[pi > 0]
    entropy = -(pi * np.log(pi)).sum()
    return float(np.exp(entropy))


def main():
    args = parse_args()
    runs = build_runs(args.ks, args.dpmm_slot)
    use_style()
    print(f"dataset: {args.dataset}")
    print(f"{'model':>10} | {'nominal K':>9} | {'active@1e-2':>11} {'active@1e-3':>11} | {'perplexity':>10}")
    print("-" * 66)

    spectra = {}
    for label, kslot, seed_folder, ModelClass in runs:
        path = os.path.join(run_dir(args.dataset, kslot, seed_folder), "checkpoint_best.pth")
        if not os.path.exists(path):
            print(f"{label:>10} | (checkpoint not found, skipped)")
            continue
        pi = load_pi(args.dataset, kslot, seed_folder, ModelClass)
        spectra[label] = pi
        a2 = int((pi > 1e-2).sum())
        a3 = int((pi > 1e-3).sum())
        print(f"{label:>10} | {len(pi):>9} | {a2:>11} {a3:>11} | {perplexity(pi):>10.2f}")

    # --- spectrum figure: sorted mixing weights per model (log-y) ---
    colors = [BLUE] + PURPLE_RAMP    # DPMM (blue) + K=10/50/86/150 (purple ramp), matches RUNS order
    fig, ax = plt.subplots(figsize=(8, 5))
    for (label, pi), color in zip(spectra.items(), colors):
        ax.plot(np.arange(1, len(pi) + 1), np.clip(pi, 1e-8, None),
                marker=".", ms=4, lw=2, color=color, label=label)
    ax.axhline(1e-2, color=MUTED, ls="--", lw=1.2, label="1% weight")
    ax.set_yscale("log")
    ax.set_xlabel("component rank (largest weight first)")
    ax.set_ylabel("mixing weight $\\pi$")
    ax.set_title(f"How many components actually carry weight? ({args.dataset})")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    os.makedirs("analysis/out", exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    np.savez(f"analysis/out/effective_components{suffix}.npz",
             **{k.replace(" ", "_").replace("=", ""): v for k, v in spectra.items()})
    fig.savefig(f"analysis/out/fig_effective_components{suffix}.png", dpi=150)
    plt.close(fig)
    print(f"\nsaved -> analysis/out/effective_components{suffix}.npz "
          f"+ fig_effective_components{suffix}.png")


if __name__ == "__main__":
    main()
