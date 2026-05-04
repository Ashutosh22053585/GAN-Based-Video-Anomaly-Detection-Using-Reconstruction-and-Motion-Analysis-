"""
train.py
========
Training loop for the GAN-Based Video Anomaly Detection system.

Training strategy:
  - The model is trained ONLY on normal videos (unsupervised anomaly detection).
  - Each iteration alternates between:
      1. Discriminator update: maximise ability to distinguish real vs reconstructed frames.
      2. Generator update:     minimise combined loss (adversarial + reconstruction + flow).
  - Checkpoints, loss curves, and sample reconstructions are saved periodically.

Run:
    python train.py
"""

import os
import time
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import config
from dataloader    import build_dataloader
from generator     import UNetGenerator
from discriminator import PatchGANDiscriminator
from loss          import AnomalyGANLoss
from utils         import (set_seed, get_device, save_checkpoint,
                            denormalize, make_grid_image)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: ONE TRAINING STEP
# ─────────────────────────────────────────────────────────────────────────────

def train_step(input_t:   torch.Tensor,
               prev_t:    torch.Tensor,
               target_t:  torch.Tensor,
               netG:      UNetGenerator,
               netD:      PatchGANDiscriminator,
               optG:      optim.Optimizer,
               optD:      optim.Optimizer,
               criterion: AnomalyGANLoss) -> dict[str, float]:
    """
    Single mini-batch training step.

    Args:
        input_t  : (B, 3, H, W) — frame + flow_u + flow_v
        target_t : (B, 1, H, W) — real frame (reconstruction target)
        netG     : generator model
        netD     : discriminator model
        optG/D   : optimizers
        criterion: combined loss object

    Returns:
        Dictionary of scalar loss values for logging.
    """
    # ── 1. Generator forward pass ─────────────────────────────────────────────
    generated = netG(input_t)   # (B, 1, H, W)

    # ── 2. Discriminator update ───────────────────────────────────────────────
    optD.zero_grad()

    # Judge real frames (detach generated to avoid G gradients flowing into D)
    pred_real = netD(input_t, target_t)
    pred_fake = netD(input_t, generated.detach())   # stop G gradient

    loss_D = criterion.discriminator_total(pred_real, pred_fake)
    loss_D.backward()
    optD.step()

    # ── 3. Generator update ───────────────────────────────────────────────────
    optG.zero_grad()

    # Re-evaluate fake WITH gradients this time (no detach)
    pred_fake_for_G = netD(input_t, generated)

    g_losses = criterion.generator_total(input_t, prev_t, generated,
                                         target_t, pred_fake_for_G)
    g_losses["total"].backward()
    optG.step()

    return {
        "loss_D":     loss_D.item(),
        "loss_G":     g_losses["total"].item(),
        "loss_G_adv": g_losses["adv"].item(),
        "loss_recon": g_losses["recon"].item(),
        "loss_flow":  g_losses["flow"].item(),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train():
    # ── Setup ─────────────────────────────────────────────────────────────────
    set_seed(config.SEED)
    device = get_device()
    print(f"[Train] Using device: {device}")

    writer = SummaryWriter(log_dir=config.LOG_DIR)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_dir = os.path.join(config.DATA_RAW_DIR, "Train")
    loader    = build_dataloader(
        train_dir,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        augment=True,
        num_workers=config.NUM_WORKERS,
    )
    print(f"[Train] Batches per epoch: {len(loader)}")

    # ── Models ────────────────────────────────────────────────────────────────
    netG = UNetGenerator().to(device)
    netD = PatchGANDiscriminator().to(device)

    # ── Optimizers ────────────────────────────────────────────────────────────
    optG = optim.Adam(netG.parameters(),
                      lr=config.LR_G,
                      betas=(config.BETA1, config.BETA2))
    optD = optim.Adam(netD.parameters(),
                      lr=config.LR_D,
                      betas=(config.BETA1, config.BETA2))

    scheduler_G = torch.optim.lr_scheduler.StepLR(optG, step_size=30, gamma=0.5)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optD, step_size=30, gamma=0.5)

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = AnomalyGANLoss(device)

    # ── Resume from checkpoint if available ───────────────────────────────────
    start_epoch  = 0
    history: dict[str, list[float]] = {
        k: [] for k in ["loss_D", "loss_G", "loss_G_adv", "loss_recon", "loss_flow"]
    }
    ckpt_path = os.path.join(config.MODEL_DIR, "latest.pth")
    if os.path.exists(ckpt_path):
        print(f"[Train] Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        netG.load_state_dict(ckpt["netG"])
        netD.load_state_dict(ckpt["netD"])
        optG.load_state_dict(ckpt["optG"])
        optD.load_state_dict(ckpt["optD"])
        start_epoch = ckpt.get("epoch", 0) + 1
        history     = ckpt.get("history", history)
        print(f"[Train] Resumed at epoch {start_epoch}")

    # ── Training epochs ───────────────────────────────────────────────────────
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        netG.train()
        netD.train()
        epoch_losses: dict[str, list[float]] = {k: [] for k in history}
        t0 = time.time()

        for batch_idx, batch in enumerate(loader):
            input_t  = batch["input"].to(device)    # (B, 3, H, W)
            prev_t   = batch["prev_frame"].to(device)
            target_t = batch["target"].to(device)   # (B, 1, H, W)

            step_losses = train_step(input_t, prev_t, target_t,
                                     netG, netD, optG, optD, criterion)

            for k, v in step_losses.items():
                epoch_losses[k].append(v)

            # ── Per-batch logging ────────────────────────────────────────────
            if (batch_idx + 1) % config.LOG_INTERVAL == 0:
                msg_parts = [f"Epoch [{epoch+1}/{config.NUM_EPOCHS}]",
                             f"Step [{batch_idx+1}/{len(loader)}]"]
                for k, v in step_losses.items():
                    msg_parts.append(f"{k}={v:.4f}")
                print("  ".join(msg_parts))

        # ── Epoch averages ────────────────────────────────────────────────────
        avg = {k: float(np.mean(v)) for k, v in epoch_losses.items()}
        elapsed = time.time() - t0
        print(f"\n[Epoch {epoch+1}] "
              f"D={avg['loss_D']:.4f}  G={avg['loss_G']:.4f}  "
              f"recon={avg['loss_recon']:.4f}  flow={avg['loss_flow']:.4f}  "
              f"({elapsed:.1f}s)\n")

        for k, v in avg.items():
            history[k].append(v)
            writer.add_scalar(f"train/{k}", v, epoch + 1)

        # ── LR scheduler step ─────────────────────────────────────────────────
        scheduler_G.step()
        scheduler_D.step()
        writer.add_scalar("train/lr_G", scheduler_G.get_last_lr()[0], epoch + 1)
        writer.add_scalar("train/lr_D", scheduler_D.get_last_lr()[0], epoch + 1)

        # ── Save checkpoint + visuals ─────────────────────────────────────────
        if (epoch + 1) % config.SAVE_INTERVAL == 0 or (epoch + 1) == config.NUM_EPOCHS:
            _save_epoch(epoch, netG, netD, optG, optD, history, device, loader)

    # ── Final loss curve ──────────────────────────────────────────────────────
    _plot_loss_curves(history)
    writer.close()
    print("[Train] Training complete ✓")


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _save_epoch(epoch, netG, netD, optG, optD, history, device, loader):
    """Save checkpoint and a grid of sample reconstructions."""
    # Checkpoint
    save_checkpoint(
        path=os.path.join(config.MODEL_DIR, f"ckpt_epoch{epoch+1:03d}.pth"),
        epoch=epoch,
        netG=netG, netD=netD, optG=optG, optD=optD, history=history,
    )
    # Also save as 'latest' for easy resuming
    save_checkpoint(
        path=os.path.join(config.MODEL_DIR, "latest.pth"),
        epoch=epoch,
        netG=netG, netD=netD, optG=optG, optD=optD, history=history,
    )

    # Visualise reconstructions for the first batch
    netG.train(False)
    with torch.no_grad():
        batch      = next(iter(loader))
        input_t    = batch["input"][:4].to(device)
        target_t   = batch["target"][:4].to(device)
        generated  = netG(input_t)

        grid_path = os.path.join(config.PLOT_DIR,
                                 f"recon_epoch{epoch+1:03d}.png")
        make_grid_image(target_t, generated, save_path=grid_path)
        print(f"[Train] Saved reconstruction grid → {grid_path}")


def _plot_loss_curves(history: dict):
    """Save a loss curve plot to the plots directory."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["loss_D"], label="D loss")
    axes[0].plot(history["loss_G"], label="G loss (total)")
    axes[0].set_title("Adversarial Losses")
    axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history["loss_recon"], label="Recon (L1)")
    axes[1].plot(history["loss_flow"],  label="Flow consistency")
    axes[1].set_title("Reconstruction & Flow Losses")
    axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    path = os.path.join(config.PLOT_DIR, "loss_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Train] Loss curves saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
