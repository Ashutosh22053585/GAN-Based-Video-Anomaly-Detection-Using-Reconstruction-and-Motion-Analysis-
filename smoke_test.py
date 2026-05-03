# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
"""
smoke_test.py
=============
Runs shape/forward-pass checks on every module WITHOUT needing real data.
Synthetic random tensors are used in place of actual frames.
All tests must pass before you start training.

Run:
    python smoke_test.py
"""

import sys
import torch
import numpy as np
import config

PASS = "[92m PASS [0m"
FAIL = "[91m FAIL [0m"
results = []

def check(name, fn):
    try:
        fn()
        print(f"[{PASS}] {name}")
        results.append((name, True))
    except Exception as e:
        print(f"[ FAIL ] {name}  ->  {e}")
        results.append((name, False))

# ── 1. Config ─────────────────────────────────────────────────────────────────
def test_config():
    import os
    assert config.IN_CHANNELS  == 3, "Expected IN_CHANNELS=3"
    assert config.OUT_CHANNELS == 1, "Expected OUT_CHANNELS=1"
    assert config.IMG_HEIGHT   == config.IMG_WIDTH == 256
    for d in [config.MODEL_DIR, config.PLOT_DIR, config.ANOMALY_MAP_DIR, config.LOG_DIR]:
        os.makedirs(d, exist_ok=True)

check("Config values & directories", test_config)

# ── 2. Optical flow (dataloader) ──────────────────────────────────────────────
def test_optical_flow():
    from dataloader import compute_optical_flow, normalize_flow
    prev = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    curr = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    flow = compute_optical_flow(prev, curr)
    assert flow.shape == (256, 256, 2), f"Bad flow shape: {flow.shape}"
    norm = normalize_flow(flow)
    assert norm.min() >= -1.0 and norm.max() <= 1.0

check("Optical flow (Farneback + normalize)", test_optical_flow)

# ── 3. Generator ──────────────────────────────────────────────────────────────
def test_generator():
    from generator import UNetGenerator
    G   = UNetGenerator()
    x   = torch.randn(2, config.IN_CHANNELS, config.IMG_HEIGHT, config.IMG_WIDTH)
    out = G(x)
    assert out.shape == (2, config.OUT_CHANNELS, config.IMG_HEIGHT, config.IMG_WIDTH), \
        f"Generator output shape: {out.shape}"
    assert out.min() >= -1.05 and out.max() <= 1.05, "Output outside [-1,1] range"

check("Generator (U-Net) forward pass", test_generator)

# ── 4. Discriminator ─────────────────────────────────────────────────────────
def test_discriminator():
    from discriminator import PatchGANDiscriminator
    D    = PatchGANDiscriminator()
    cond = torch.randn(2, config.IN_CHANNELS,  config.IMG_HEIGHT, config.IMG_WIDTH)
    tgt  = torch.randn(2, config.OUT_CHANNELS, config.IMG_HEIGHT, config.IMG_WIDTH)
    out  = D(cond, tgt)
    assert out.shape[0] == 2 and out.shape[1] == 1, \
        f"Discriminator output: {out.shape}"
    assert out.shape[2] > 1 and out.shape[3] > 1, "Expected 2D patch map, got scalar"

check("Discriminator (PatchGAN) forward pass", test_discriminator)

# ── 5. Loss functions ─────────────────────────────────────────────────────────
def test_loss():
    from loss import AnomalyGANLoss
    device    = torch.device("cpu")
    criterion = AnomalyGANLoss(device)
    B, H, W   = 2, config.IMG_HEIGHT, config.IMG_WIDTH

    inp  = torch.randn(B, config.IN_CHANNELS,  H, W)
    gen  = torch.randn(B, config.OUT_CHANNELS, H, W)
    tgt  = torch.randn(B, config.OUT_CHANNELS, H, W)
    fake = torch.randn(B, 1, 30, 30)
    real = torch.randn(B, 1, 30, 30)

    g = criterion.generator_total(inp, gen, tgt, fake)
    d = criterion.discriminator_total(real, fake)

    assert "total" in g and "adv" in g and "recon" in g and "flow" in g
    assert g["total"].item() > 0
    assert d.item() > 0

check("Loss functions (LSGAN + L1 + Flow)", test_loss)

# ── 6. Full forward pass (G → D → Losses) ────────────────────────────────────
def test_full_pipeline():
    from generator     import UNetGenerator
    from discriminator import PatchGANDiscriminator
    from loss          import AnomalyGANLoss

    device = torch.device("cpu")
    G  = UNetGenerator().to(device)
    D  = PatchGANDiscriminator().to(device)
    L  = AnomalyGANLoss(device)

    B  = 2
    inp = torch.randn(B, config.IN_CHANNELS,  config.IMG_HEIGHT, config.IMG_WIDTH)
    tgt = torch.randn(B, config.OUT_CHANNELS, config.IMG_HEIGHT, config.IMG_WIDTH)

    gen       = G(inp)
    pred_real = D(inp, tgt)
    pred_fake = D(inp, gen.detach())

    g_losses = L.generator_total(inp, gen, tgt, D(inp, gen))
    d_loss   = L.discriminator_total(pred_real, pred_fake)

    g_losses["total"].backward()
    d_loss.backward()

check("End-to-end pipeline (G->D->loss->backward)", test_full_pipeline)

# ── 7. Utils ──────────────────────────────────────────────────────────────────
def test_utils():
    from utils import set_seed, get_device, denormalize, normalize_scores
    set_seed(42)
    dev = get_device()
    assert dev is not None

    t = torch.tensor([[[-0.5, 0.5], [1.0, -1.0]]])
    arr = denormalize(t)
    assert arr.dtype == np.uint8
    assert arr.shape == (2, 2)

    scores  = np.array([0.1, 0.5, 1.0, 0.2])
    normed  = normalize_scores(scores)
    assert abs(normed.max() - 1.0) < 1e-5

check("Utils (seed, device, denormalize, normalize_scores)", test_utils)

# ── Final report ──────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
passed = sum(1 for _, ok in results if ok)
total  = len(results)
print(f"  Results: {passed}/{total} tests passed")
if passed == total:
    print("  ALL TESTS PASSED -- Ready to train!")
else:
    failed = [n for n, ok in results if not ok]
    print("  FAILED tests:", ", ".join(failed))
print("=" * 50)

sys.exit(0 if passed == total else 1)
