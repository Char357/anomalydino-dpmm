"""Smoke test for the DINOv3 backbone.

Verifies the model downloads, loads through our DINOv3Wrapper, and produces patch features of
the right shape -- BEFORE committing an a100 to a full run. Run on the LOGIN node (needs internet
+ your HF token; no GPU needed):

    python dinov3_smoketest.py
"""
import sys
import torch

print("torch:", torch.__version__)
try:
    import transformers
    print("transformers:", transformers.__version__)
except ImportError:
    print("!! transformers is NOT installed in this env. Install it (login node):")
    print("     pip install -U transformers")
    sys.exit(1)

from src.backbones import get_model

print("\nloading dinov3_vits16 (first run downloads ~90 MB to the HF cache)...")
try:
    m = get_model("dinov3_vits16", device="cpu", smaller_edge_size=448)
except Exception as e:
    print("\n!! load failed:", repr(e))
    print("   - gated download? run  huggingface-cli login  (paste your token), license already accepted")
    print("   - unknown model 'dinov3'? your transformers is too old:  pip install -U transformers")
    sys.exit(1)

print("OK. embedding dim:", m.get_embedding_dimension())

# crop to a multiple of the patch size (16 -> 28x28 grid at 448)
cropped, grid = m.crop_image(torch.randn(1, 3, 450, 450))
print("crop_image ->", tuple(cropped.shape), "grid:", grid)

# extract patch features and check the token count matches the grid
feats = m.extract_features(cropped, use_pca=False)
n_expected = grid[0] * grid[1]
print("patch features:", tuple(feats.shape), "| expected tokens:", n_expected)

ok = feats.shape[1] == n_expected and feats.shape[2] == m.get_embedding_dimension()
print("\nSMOKE TEST PASSED" if ok else "\n!! SHAPE MISMATCH -- check register-token stripping")
sys.exit(0 if ok else 1)
