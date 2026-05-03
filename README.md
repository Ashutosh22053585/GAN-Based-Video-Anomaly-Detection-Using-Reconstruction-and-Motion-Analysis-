# GAN-Based Video Anomaly Detection — Using Reconstruction and Motion Analysis

## Overview
An **unsupervised** deep learning system that detects anomalous events in surveillance videos by learning to reconstruct *only normal frames*. At test time, frames the model cannot reconstruct well are flagged as anomalies.

---

## Architecture

```
Input: (frame_t, optical_flow_t)          3 channels: gray + u-flow + v-flow
          │
          ▼
  ┌─────────────────┐
  │  U-Net Generator│  ← trained only on normal videos
  └────────┬────────┘
           │ reconstructed_frame
           ▼
  ┌─────────────────────┐
  │ PatchGAN Discriminator│  ← judges patches as real or reconstructed
  └─────────────────────┘

Loss = λ_adv × LSGAN_loss + λ_recon × L1_loss + λ_flow × Flow_consistency_loss

Anomaly Score = mean pixel-wise MSE between original and reconstructed frame
```

---

## Project Structure

```
Gan based project/
├── config.py          # All hyper-parameters and paths
├── dataloader.py      # Frame loading + Farneback optical flow + Dataset class
├── generator.py       # U-Net Generator (encoder-decoder with skip connections)
├── discriminator.py   # PatchGAN Discriminator (InstanceNorm + Conv2d blocks)
├── loss.py            # LSGAN + L1 Reconstruction + Flow Consistency losses
├── train.py           # Full training loop (alternating G/D updates)
├── evaluate.py        # Inference, anomaly scoring, AUC-ROC, heat-maps
├── utils.py           # Seed, device, checkpointing, visualisation helpers
├── requirements.txt   # Python dependencies
│
├── data/
│   ├── raw/           # ← Place your dataset here
│   │   ├── Train/
│   │   │   ├── video_001/  (000.tif, 001.tif, ...)
│   │   │   └── video_002/
│   │   └── Test/
│   │       ├── video_001/
│   │       └── ...
│   └── processed/     # (optional) cached tensors
│
├── models/            # Saved checkpoints (.pth)
├── outputs/
│   ├── plots/         # Loss curves, reconstruction grids, score plots
│   └── anomaly_maps/  # Per-frame heat-map PNGs
└── logs/              # TensorBoard event files
```

---

## Dataset Setup

### UCSD Pedestrian Dataset (recommended)
1. Download from: http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm
2. Extract so that the directory tree matches `data/raw/Train/` and `data/raw/Test/`.

### CUHK Avenue Dataset (backup)
1. Download from: http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html
2. Use `utils.extract_frames()` to convert `.avi` clips to PNG frame folders.

```python
from utils import extract_frames
extract_frames("Avenue/train/01.avi", "data/raw/Train/clip_01/")
```

---

## Installation

```bash
pip install -r requirements.txt
```

For CUDA 12.x:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Training

```bash
python train.py
```

- Trains only on `data/raw/Train/` (normal frames only).
- Saves checkpoints every `SAVE_INTERVAL` epochs to `models/`.
- Logs losses and LR to TensorBoard:

```bash
tensorboard --logdir logs/
```

---

## Evaluation / Inference

```bash
python evaluate.py \
    --checkpoint models/ckpt_epoch050.pth \
    --test_dir   data/raw/Test \
    [--gt_dir    data/raw/Test_gt] \
    [--threshold 0.5]
```

Outputs:
- Per-frame anomaly heat-maps → `outputs/anomaly_maps/<clip>/`
- Score-vs-time plots         → `outputs/plots/<clip>_scores.png`
- AUC-ROC printed to console (if `--gt_dir` provided)

---

## Key Configuration (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMG_HEIGHT/WIDTH` | 256 | Resize target for all frames |
| `BATCH_SIZE` | 16 | Training mini-batch size |
| `NUM_EPOCHS` | 50 | Total training epochs |
| `LR_G / LR_D` | 2e-4 | Learning rates |
| `LAMBDA_RECON` | 10.0 | L1 reconstruction loss weight |
| `LAMBDA_FLOW` | 5.0 | Flow consistency loss weight |
| `ANOMALY_THRESHOLD` | 0.5 | MSE threshold for binary prediction |
| `SMOOTHING_WINDOW` | 5 | Temporal smoothing window (frames) |
| `NGF / NDF` | 64 | Generator / Discriminator base filters |

---

## How Anomaly Detection Works (Test Time)

```
For each frame pair (prev_frame, curr_frame):
  1. Compute optical flow  →  (u, v) channels
  2. Build input tensor    →  [curr_frame, u, v]  (3 × 256 × 256)
  3. Generator reconstructs the frame
  4. Compute pixel-wise MSE  →  anomaly score map
  5. Average score map       →  scalar anomaly score
  6. Apply temporal smoothing
  7. score > threshold  →  ANOMALY flag
```

**Intuition:** The generator learned to reconstruct normal patterns efficiently.
Anomalous patterns (running, fighting, unusual objects) produce higher errors
because they were never seen during training.

---

## Citation / References

- Goodfellow et al., "Generative Adversarial Networks" (2014)
- Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks" — pix2pix (2017)
- Lim & Hua, "Abnormal Event Detection at 150 FPS in MATLAB" (2015)
- UCSD Anomaly Detection Dataset — Mahadevan et al. (2010)
