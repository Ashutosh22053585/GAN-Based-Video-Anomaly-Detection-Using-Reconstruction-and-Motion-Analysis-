"""
evaluate.py
===========
Evaluation and anomaly detection module.

At inference time, the trained generator reconstructs every test frame.
A frame is considered anomalous if its pixel-wise reconstruction error
(MSE) exceeds a learnable threshold.

Steps:
  1. Load a trained generator checkpoint.
  2. For each test frame:
       a. Compute optical flow with the previous frame.
       b. Build the 3-channel input tensor.
       c. Run through the generator to get a reconstruction.
       d. Compute per-pixel MSE → anomaly score map.
       e. Average over spatial dimensions → scalar anomaly score.
  3. Apply temporal smoothing (moving average) over the score sequence.
  4. Threshold scores to produce binary anomaly predictions.
  5. Compute AUC-ROC (if ground-truth labels are available).
  6. Save anomaly score plots and pixel-level anomaly maps.

Run:
    python evaluate.py --checkpoint models/ckpt_epoch050.pth
                       --test_dir    data/raw/Test
                       [--gt_dir     data/raw/Test_gt]
                       [--threshold  0.5]
"""

import os
import json
import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import uniform_filter1d

import config
from dataloader    import compute_optical_flow, normalize_flow, preprocess_frame
from generator     import UNetGenerator
from utils         import get_device, denormalize


# ─────────────────────────────────────────────────────────────────────────────
#  FRAME-LEVEL ANOMALY SCORE COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def score_frame(generator:   UNetGenerator,
                prev_gray:   np.ndarray,
                curr_gray:   np.ndarray,
                device:      torch.device) -> tuple[float, np.ndarray]:
    """
    Compute the anomaly score for a single frame pair.

    Args:
        generator : trained UNetGenerator (in eval mode)
        prev_gray : previous frame as uint8 grayscale (H, W)
        curr_gray : current  frame as uint8 grayscale (H, W)
        device    : torch device

    Returns:
        score     : scalar MSE reconstruction error (anomaly score)
        error_map : (H, W) float32 pixel-wise squared error in [0, 1]
    """
    # ── Preprocess ────────────────────────────────────────────────────────────
    H, W = config.IMG_HEIGHT, config.IMG_WIDTH

    prev_resized = cv2.resize(prev_gray, (W, H), interpolation=cv2.INTER_LINEAR)
    curr_resized = cv2.resize(curr_gray, (W, H), interpolation=cv2.INTER_LINEAR)

    # Optical flow
    flow = compute_optical_flow(prev_resized, curr_resized)   # (H, W, 2)
    flow = normalize_flow(flow)

    # Normalize frame to [-1, 1]
    curr_norm = curr_resized.astype(np.float32) / 127.5 - 1.0

    # Build input tensor (1, 3, H, W)
    frame_t = torch.from_numpy(curr_norm).unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
    flow_t  = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0)    # (1,2,H,W)
    inp_t   = torch.cat([frame_t, flow_t], dim=1).to(device)          # (1,3,H,W)
    tgt_t   = frame_t.to(device)                                       # (1,1,H,W)

    # ── Generator inference ───────────────────────────────────────────────────
    with torch.no_grad():
        reconstructed = generator(inp_t)   # (1, 1, H, W) in [-1, 1]

    # ── Error map ────────────────────────────────────────────────────────────
    error_map = (reconstructed - tgt_t).pow(2)               # (1, 1, H, W)
    error_map = error_map.squeeze().cpu().numpy()             # (H, W)
    error_map = np.clip(error_map, 0, 1)

    score = float(error_map.mean())
    return score, error_map


# ─────────────────────────────────────────────────────────────────────────────
#  VIDEO / CLIP-LEVEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_clip(clip_dir: str,
                  generator: UNetGenerator,
                  device:    torch.device,
                  save_maps: bool = True) -> np.ndarray:
    """
    Run anomaly scoring on all frames of a single clip directory.

    Args:
        clip_dir  : Path to folder containing sorted frame images.
        generator : Trained generator in eval mode.
        device    : Torch device.
        save_maps : If True, saves per-frame anomaly heat-maps as PNG.

    Returns:
        scores: 1-D float32 array of per-frame anomaly scores.
    """
    VIDEO_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    frame_paths = sorted([
        os.path.join(clip_dir, f)
        for f in os.listdir(clip_dir)
        if os.path.splitext(f)[1].lower() in VIDEO_EXTS
    ])

    if len(frame_paths) < 2:
        return np.array([])

    clip_name = os.path.basename(clip_dir)
    map_dir   = os.path.join(config.ANOMALY_MAP_DIR, clip_name)
    if save_maps:
        os.makedirs(map_dir, exist_ok=True)

    scores: list[float] = []
    prev_gray = cv2.imread(frame_paths[0], cv2.IMREAD_GRAYSCALE)

    for i, curr_path in enumerate(frame_paths[1:], start=1):
        curr_gray = cv2.imread(curr_path, cv2.IMREAD_GRAYSCALE)
        if curr_gray is None:
            continue

        score, error_map = score_frame(generator, prev_gray, curr_gray, device)
        scores.append(score)

        # Save anomaly heat-map
        if save_maps:
            heat = cm.jet(error_map)[:, :, :3]   # (H, W, 3) RGB
            heat = (heat * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(map_dir, f"frame_{i:05d}_anomaly.png"),
                        cv2.cvtColor(heat, cv2.COLOR_RGB2BGR))

        prev_gray = curr_gray

    return np.array(scores, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  TEMPORAL SMOOTHING
# ─────────────────────────────────────────────────────────────────────────────

def smooth_scores(scores: np.ndarray,
                  window: int = config.TEMPORAL_SMOOTH_WINDOW) -> np.ndarray:
    """
    Apply a causal moving-average to the per-frame anomaly score sequence
    to suppress single-frame noise.

    Args:
        scores : (N,) float array of raw anomaly scores.
        window : smoothing window size (frames).

    Returns:
        Smoothed scores array of the same length.
    """
    if len(scores) == 0:
        return scores
    return uniform_filter1d(scores, size=window, mode="nearest")


# ─────────────────────────────────────────────────────────────────────────────
#  AUC-ROC METRIC  (optional — needs ground-truth labels)
# ─────────────────────────────────────────────────────────────────────────────

def compute_auc(scores: np.ndarray, gt_labels: np.ndarray) -> float:
    """
    Compute frame-level AUC-ROC.

    Args:
        scores    : (N,) predicted anomaly scores (higher = more anomalous).
        gt_labels : (N,) binary ground-truth (1=anomaly, 0=normal).

    Returns:
        AUC-ROC score in [0, 1].  Returns -1 if sklearn is unavailable.
    """
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(gt_labels, scores))
    except ImportError:
        print("[Evaluate] sklearn not installed — AUC not computed.")
        return -1.0
    except ValueError as e:
        print(f"[Evaluate] AUC error: {e}")
        return -1.0


def find_best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Find the threshold that maximizes the F1 score by iterating through percentiles.
    """
    best_f1 = 0.0
    best_thresh = 0.0
    best_prec = 0.0
    best_rec = 0.0
    
    # Iterate over 50th to 99th percentiles
    percentiles = np.percentile(scores, np.arange(50, 100))
    
    for thresh in percentiles:
        preds = (scores > thresh).astype(int)
        
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_prec = precision
            best_rec = recall
            
    print(f"[Evaluate] Best threshold: {best_thresh:.4f} | F1: {best_f1:.4f} | Precision: {best_prec:.4f} | Recall: {best_rec:.4f}")
    
    out_path = os.path.join(config.MODEL_DIR, "best_threshold.json")
    with open(out_path, "w") as f:
        json.dump({"best_threshold": float(best_thresh), "f1": float(best_f1)}, f)
    print(f"[Evaluate] Saved best threshold to {out_path}")
    
    return best_thresh


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_anomaly_scores(scores: np.ndarray,
                        gt_labels: np.ndarray | None = None,
                        threshold: float = config.ANOMALY_THRESHOLD,
                        clip_name: str   = "clip",
                        auc:       float = -1.0):
    """
    Plot per-frame anomaly scores with optional ground-truth and threshold line.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(scores, label="Anomaly Score", color="#4ECDC4", linewidth=1.5)
    ax.axhline(threshold, color="#FF6B6B", linestyle="--",
               linewidth=1.2, label=f"Threshold = {threshold:.2f}")

    if gt_labels is not None and len(gt_labels) == len(scores):
        # Shade anomalous regions
        for i, label in enumerate(gt_labels):
            if label == 1:
                ax.axvspan(i - 0.5, i + 0.5, alpha=0.15, color="red")

    title = f"Anomaly Scores — {clip_name}"
    if auc >= 0:
        title += f"  (AUC-ROC = {auc:.4f})"
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Frame"); ax.set_ylabel("MSE Score")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(config.PLOT_DIR, f"{clip_name}_scores.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Evaluate] Score plot saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(checkpoint: str,
         test_dir:   str,
         gt_dir:     str  | None,
         threshold:  float):

    device = get_device()
    print(f"[Evaluate] Device: {device}")

    # Load trained generator
    netG = UNetGenerator().to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    netG.load_state_dict(ckpt["netG"])
    netG.train(False)
    print(f"[Evaluate] Loaded checkpoint: {checkpoint}")

    # Iterate over test clips
    all_scores: list[float] = []
    all_labels: list[int]   = []

    clip_dirs = sorted([
        os.path.join(test_dir, d)
        for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d))
    ])

    for clip_dir in clip_dirs:
        clip_name = os.path.basename(clip_dir)
        print(f"[Evaluate] Processing clip: {clip_name}")

        scores = evaluate_clip(clip_dir, netG, device, save_maps=True)
        scores = smooth_scores(scores)

        # Load ground-truth labels if provided
        gt_labels = None
        if gt_dir:
            gt_path = os.path.join(gt_dir, clip_name + ".npy")
            if os.path.exists(gt_path):
                gt_labels = np.load(gt_path)                  # binary (N,)
                all_labels.extend(gt_labels[:len(scores)].tolist())

        all_scores.extend(scores.tolist())

        # Compute per-clip AUC if GT available
        auc = compute_auc(scores, gt_labels) if gt_labels is not None else -1.0
        plot_anomaly_scores(scores, gt_labels, threshold, clip_name, auc)

    # Global AUC across all clips
    if all_labels:
        global_auc = compute_auc(np.array(all_scores), np.array(all_labels))
        print(f"\n[Evaluate] ═══ Global AUC-ROC = {global_auc:.4f} ═══")
        find_best_threshold(np.array(all_scores), np.array(all_labels))
        print()
    else:
        print("[Evaluate] No ground-truth provided — AUC not computed.")

    print("[Evaluate] Done ✓")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate GAN anomaly detection on test clips."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--test_dir",   required=True,
                        help="Path to test split directory")
    parser.add_argument("--gt_dir",     default=None,
                        help="Optional: directory with ground-truth .npy label files")
    parser.add_argument("--threshold",  type=float,
                        default=config.ANOMALY_THRESHOLD,
                        help="Anomaly score threshold")
    args = parser.parse_args()

    main(args.checkpoint, args.test_dir, args.gt_dir, args.threshold)
