"""
dataloader.py
=============
Handles all data ingestion for the GAN-Based Video Anomaly Detection system.

Pipeline per sample:
  1. Load a grayscale frame  (H × W × 1)
  2. Load / compute optical flow between that frame and the previous one (H × W × 2)
  3. Resize both to IMG_HEIGHT × IMG_WIDTH
  4. Normalize to [-1, 1]
  5. Concatenate → tensor of shape (IN_CHANNELS, H, W) = (3, 256, 256)

Optical Flow:
  Uses Gunnar Farneback's dense optical flow (cv2.calcOpticalFlowFarneback).
  For the very first frame in a sequence the flow is zero-filled.

Expected dataset folder layout (UCSD / CUHK Avenue):
  data/raw/
    Train/
      video_001/
        000.tif (or .jpg / .png)
        001.tif
        ...
      video_002/
        ...
    Test/
      video_001/
        ...
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import config


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _load_frame_paths(root_dir: str) -> list[tuple[str, list[str]]]:
    """
    Walk root_dir and collect (video_folder, [sorted frame paths]) tuples.
    Supports .tif / .png / .jpg / .bmp images.
    """
    VIDEO_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    clips = []
    for clip_name in sorted(os.listdir(root_dir)):
        clip_path = os.path.join(root_dir, clip_name)
        if not os.path.isdir(clip_path):
            continue
        frames = sorted(
            [os.path.join(clip_path, f) for f in os.listdir(clip_path)
             if os.path.splitext(f)[1].lower() in VIDEO_EXTS]
        )
        if len(frames) >= 2:          # need at least 2 frames for flow
            clips.append((clip_name, frames))
    return clips


def compute_optical_flow(prev_gray: np.ndarray,
                          curr_gray: np.ndarray) -> np.ndarray:
    """
    Compute dense optical flow using Gunnar Farneback's method.

    Args:
        prev_gray: Grayscale uint8 image (H, W)
        curr_gray: Grayscale uint8 image (H, W)

    Returns:
        flow: float32 ndarray of shape (H, W, 2)
              flow[..., 0] = horizontal component (u)
              flow[..., 1] = vertical  component (v)
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,   # image pyramid scale (< 1 to down-sample)
        levels=3,         # number of pyramid levels
        winsize=15,       # averaging window size
        iterations=3,     # iterations per pyramid level
        poly_n=5,         # pixel neighbourhood size for polynomial expansion
        poly_sigma=1.2,   # Gaussian σ for polynomial expansion
        flags=0
    )
    return flow.astype(np.float32)   # (H, W, 2)


def normalize_flow(flow: np.ndarray,
                   clip_val: float = 20.0) -> np.ndarray:
    """
    Normalize optical flow to [-1, 1] by clipping to ±clip_val then dividing.

    Args:
        flow:     (H, W, 2) float32 array
        clip_val: flow values are clipped to [-clip_val, clip_val]

    Returns:
        Normalized flow in [-1, 1], shape (H, W, 2)
    """
    flow = np.clip(flow, -clip_val, clip_val) / clip_val
    return flow


def preprocess_frame(bgr_or_gray: np.ndarray,
                     target_h: int = config.IMG_HEIGHT,
                     target_w: int = config.IMG_WIDTH) -> np.ndarray:
    """
    Resize to target resolution and convert to float32 in [-1, 1].

    Args:
        bgr_or_gray: uint8 image, can be (H, W) or (H, W, 3)
        target_h/w:  resize dimensions

    Returns:
        frame: float32 ndarray of shape (target_h, target_w)
    """
    # Convert to grayscale if needed
    if bgr_or_gray.ndim == 3 and bgr_or_gray.shape[2] == 3:
        gray = cv2.cvtColor(bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = bgr_or_gray

    resized = cv2.resize(gray, (target_w, target_h),
                         interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 127.5 - 1.0   # [0,255] → [-1,1]
    return normalized


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET CLASS
# ─────────────────────────────────────────────────────────────────────────────

class VideoAnomalyDataset(Dataset):
    """
    Frame-level dataset for GAN-based anomaly detection.

    Each item is a dict:
        'input'  : FloatTensor (IN_CHANNELS, H, W)   frame + flow concatenated
        'target' : FloatTensor (OUT_CHANNELS, H, W)  original frame (reconstruction target)
        'frame_path': str                             path to current frame (for debugging)

    Args:
        root_dir (str): Path to Train or Test split directory.
        stride   (int): Sample every `stride`-th frame to reduce dataset size.
        augment  (bool): Apply random horizontal flip during training.
    """

    def __init__(self, root_dir: str,
                 stride: int = config.FRAME_STRIDE,
                 augment: bool = False):
        self.augment = augment
        self.samples = []   # list of (prev_path, curr_path)

        clips = _load_frame_paths(root_dir)
        if not clips:
            raise FileNotFoundError(
                f"No video clip sub-folders found in: {root_dir}\n"
                "Ensure your dataset is extracted to data/raw/Train/ "
                "with one sub-folder per clip."
            )

        for _, frame_paths in clips:
            # Stride sampling: include frame pairs at intervals
            for i in range(1, len(frame_paths), stride):
                self.samples.append((frame_paths[i - 1], frame_paths[i]))

        print(f"[DataLoader] Loaded {len(self.samples)} frame-pairs "
              f"from {len(clips)} clips in '{root_dir}'")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_gray_uint8(self, path: str) -> np.ndarray:
        """Load image as grayscale uint8 (H, W)."""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Cannot read image: {path}")
        return img

    def _maybe_augment(self, prev_norm: np.ndarray, curr_norm: np.ndarray,
                       flow: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Random horizontal flip and brightness jitter."""
        if not self.augment:
            return prev_norm, curr_norm, flow

        # Horizontal flip
        if np.random.rand() > 0.5:
            prev_norm = np.fliplr(prev_norm).copy()
            curr_norm = np.fliplr(curr_norm).copy()
            flow  = np.fliplr(flow).copy()
            flow[..., 0] *= -1   # flip u-component sign
            
        # Brightness jitter +/- 10%
        factor = np.random.uniform(0.9, 1.1)
        prev_norm = np.clip((prev_norm + 1.0) * factor - 1.0, -1.0, 1.0)
        curr_norm = np.clip((curr_norm + 1.0) * factor - 1.0, -1.0, 1.0)

        return prev_norm, curr_norm, flow

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        prev_path, curr_path = self.samples[idx]

        # --- 1. Load raw grayscale frames ---
        prev_gray_u8 = self._load_gray_uint8(prev_path)
        curr_gray_u8 = self._load_gray_uint8(curr_path)

        # --- 2. Resize both frames to target resolution BEFORE flow ---
        #        (keeps flow consistent with the model's spatial resolution)
        prev_resized = cv2.resize(prev_gray_u8,
                                  (config.IMG_WIDTH, config.IMG_HEIGHT),
                                  interpolation=cv2.INTER_LINEAR)
        curr_resized = cv2.resize(curr_gray_u8,
                                  (config.IMG_WIDTH, config.IMG_HEIGHT),
                                  interpolation=cv2.INTER_LINEAR)

        # --- 3. Compute optical flow on resized uint8 frames ---
        flow = compute_optical_flow(prev_resized, curr_resized)  # (H, W, 2)
        flow = normalize_flow(flow)                              # [-1, 1]

        # --- 4. Normalize frames to [-1, 1] ---
        prev_norm = prev_resized.astype(np.float32) / 127.5 - 1.0  # (H, W)
        curr_norm = curr_resized.astype(np.float32) / 127.5 - 1.0  # (H, W)

        # --- 5. Optional augmentation ---
        prev_norm, curr_norm, flow = self._maybe_augment(prev_norm, curr_norm, flow)

        # --- 6. Build tensors ---
        #   frame:  (1, H, W)
        #   flow:   (2, H, W)
        #   input:  (3, H, W)  ← concatenated along channel dim
        prev_tensor  = torch.from_numpy(prev_norm).unsqueeze(0)  # (1, H, W)
        frame_tensor = torch.from_numpy(curr_norm).unsqueeze(0)  # (1, H, W)
        flow_tensor  = torch.from_numpy(flow).permute(2, 0, 1)   # (2, H, W)
        input_tensor = torch.cat([frame_tensor, flow_tensor], dim=0)  # (3, H, W)

        return {
            "input":      input_tensor,           # (3, H, W) in [-1, 1]
            "target":     frame_tensor,           # (1, H, W) in [-1, 1]
            "prev_frame": prev_tensor,            # (1, H, W) in [-1, 1]
            "frame_path": curr_path,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  DATALOADER FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloader(root_dir: str,
                     batch_size: int = config.BATCH_SIZE,
                     shuffle: bool = True,
                     augment: bool = False,
                     num_workers: int = config.NUM_WORKERS) -> DataLoader:
    """
    Convenience factory that returns a ready-to-use DataLoader.

    Args:
        root_dir    : Dataset split directory (Train/ or Test/).
        batch_size  : Mini-batch size.
        shuffle     : Shuffle samples each epoch (True for training).
        augment     : Enable augmentation (True for training only).
        num_workers : Parallel loading workers.

    Returns:
        torch.utils.data.DataLoader
    """
    dataset = VideoAnomalyDataset(root_dir, augment=augment)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,       # speed up GPU transfer
        drop_last=True,        # keep batch sizes uniform during training
    )
    return loader


# ─────────────────────────────────────────────────────────────────────────────
#  QUICK SMOKE-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    train_dir = os.path.join(config.DATA_RAW_DIR, "Train")
    loader    = build_dataloader(train_dir, batch_size=4, shuffle=False)

    batch = next(iter(loader))
    inp  = batch["input"]    # (B, 3, H, W)
    tgt  = batch["target"]   # (B, 1, H, W)

    print(f"Input tensor shape : {inp.shape}")   # expect (4, 3, 256, 256)
    print(f"Target tensor shape: {tgt.shape}")   # expect (4, 1, 256, 256)
    print(f"Input  value range : [{inp.min():.2f}, {inp.max():.2f}]")
    print(f"Target value range : [{tgt.min():.2f}, {tgt.max():.2f}]")

    # Visualise: frame | u-flow | v-flow
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    titles = ["Frame (norm)", "Flow-u", "Flow-v"]
    for col in range(4):
        for row, (ch, title) in enumerate(zip([0, 1, 2], titles)):
            axes[row, col].imshow(inp[col, ch].numpy(), cmap="gray")
            axes[row, col].set_title(f"Sample {col} | {title}", fontsize=8)
            axes[row, col].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOT_DIR, "dataloader_smoke_test.png"), dpi=120)
    plt.show()
    print("[DataLoader] Smoke test complete — plot saved.")
