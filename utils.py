"""
utils.py
========
Shared utility functions used across train.py, evaluate.py, and notebooks.

Covers:
  - Reproducibility (seed setting)
  - Device selection
  - Checkpoint save / load
  - Tensor ↔ image conversions
  - Reconstruction grid visualisation
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import config


# ─────────────────────────────────────────────────────────────────────────────
#  REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = config.SEED):
    """
    Set random seeds for Python, NumPy, and PyTorch for reproducible runs.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN (may slow training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
#  DEVICE SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Return the best available device (CUDA > CPU).
    Respects config.DEVICE but falls back gracefully if CUDA is unavailable.
    """
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[Utils] CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        if config.DEVICE == "cuda":
            print("[Utils] CUDA requested but not available — using CPU.")
    return dev


# ─────────────────────────────────────────────────────────────────────────────
#  CHECKPOINT I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(path: str, epoch: int, netG, netD, optG, optD,
                    history: dict):
    """
    Save model and optimizer states to disk.

    Args:
        path    : Destination .pth file path.
        epoch   : Current epoch index (0-based).
        netG/D  : Generator / Discriminator models.
        optG/D  : Their optimizers.
        history : Dict of loss lists.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":   epoch,
        "netG":    netG.state_dict(),
        "netD":    netD.state_dict(),
        "optG":    optG.state_dict(),
        "optD":    optD.state_dict(),
        "history": history,
    }, path)
    print(f"[Utils] Checkpoint saved → {path}")


def load_generator(checkpoint_path: str,
                   device: torch.device) -> "UNetGenerator":
    """
    Load only the generator from a checkpoint for inference.

    Args:
        checkpoint_path: Path to .pth checkpoint.
        device         : Device to map the model to.

    Returns:
        Generator model in eval mode.
    """
    from generator import UNetGenerator
    netG = UNetGenerator().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    netG.load_state_dict(ckpt["netG"])
    netG.train(False)
    return netG


# ─────────────────────────────────────────────────────────────────────────────
#  TENSOR / IMAGE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized tensor in [-1, 1] to a uint8 image array in [0, 255].

    Args:
        tensor: (1, H, W) or (H, W) float tensor in [-1, 1].

    Returns:
        (H, W) uint8 numpy array.
    """
    arr = tensor.squeeze().cpu().numpy()       # (H, W)
    arr = (arr + 1.0) / 2.0                   # [-1,1] → [0,1]
    arr = np.clip(arr * 255.0, 0, 255)
    return arr.astype(np.uint8)


def make_grid_image(real_batch:      torch.Tensor,
                    generated_batch: torch.Tensor,
                    save_path:       str,
                    max_samples:     int = 4):
    """
    Create a comparison grid of real vs. reconstructed frames and save as PNG.

    Layout: two rows
        Row 1: real frames
        Row 2: generated (reconstructed) frames

    Args:
        real_batch      : (B, 1, H, W) tensor — ground-truth frames.
        generated_batch : (B, 1, H, W) tensor — generator outputs.
        save_path       : Output PNG path.
        max_samples     : Maximum number of frame pairs to display.
    """
    n = min(real_batch.size(0), max_samples)
    fig = plt.figure(figsize=(n * 3, 6))
    gs  = gridspec.GridSpec(2, n, hspace=0.05, wspace=0.05)

    for col in range(n):
        # Row 0: real
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(denormalize(real_batch[col]), cmap="gray", vmin=0, vmax=255)
        ax.set_title("Real" if col == 0 else "", fontsize=9)
        ax.axis("off")

        # Row 1: reconstructed
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(denormalize(generated_batch[col]), cmap="gray", vmin=0, vmax=255)
        ax.set_title("Recon" if col == 0 else "", fontsize=9)
        ax.axis("off")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  ANOMALY SCORE NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """
    Min-max normalise an array of anomaly scores to [0, 1].

    Useful for comparing across clips with different absolute error magnitudes.

    Args:
        scores: 1-D float array.

    Returns:
        Normalised array in [0, 1].
    """
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-8:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)


# ─────────────────────────────────────────────────────────────────────────────
#  FRAME EXTRACTION HELPER  (for raw .avi / .mp4 videos)
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(video_path: str,
                   out_dir:    str,
                   stride:     int = 1,
                   gray:       bool = True):
    """
    Extract frames from a video file and save as PNG images.

    Useful if the dataset is distributed as .avi files rather than
    pre-extracted frame folders.

    Args:
        video_path : Path to input video file.
        out_dir    : Directory to write extracted frames.
        stride     : Save every `stride`-th frame.
        gray       : Convert frames to grayscale before saving.
    """
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frame_idx, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            if gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fname = os.path.join(out_dir, f"{saved:06d}.png")
            cv2.imwrite(fname, frame)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"[Utils] Extracted {saved} frames from '{video_path}' → '{out_dir}'")
