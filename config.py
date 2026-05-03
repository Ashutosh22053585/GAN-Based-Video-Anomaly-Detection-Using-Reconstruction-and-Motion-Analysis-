"""
config.py
=========
Central configuration for the GAN-Based Video Anomaly Detection project.
All hyper-parameters, paths, and training settings are defined here so
every other module can import them from a single source of truth.
"""

import os

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")          # original videos / frame folders
DATA_PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")    # cached flow tensors (optional)
MODEL_DIR       = os.path.join(BASE_DIR, "models")               # saved checkpoints
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")
PLOT_DIR        = os.path.join(OUTPUT_DIR, "plots")
ANOMALY_MAP_DIR = os.path.join(OUTPUT_DIR, "anomaly_maps")
LOG_DIR         = os.path.join(BASE_DIR, "logs")

# ─────────────────────────────────────────────
#  IMAGE / FRAME SETTINGS
# ─────────────────────────────────────────────
IMG_HEIGHT  = 256       # resize target height
IMG_WIDTH   = 256       # resize target width
IMG_CHANNELS = 1        # 1 = grayscale, 3 = RGB
FLOW_CHANNELS = 2       # optical flow has 2 components (u, v)

# Total input channels fed to the generator = frame channels + flow channels
IN_CHANNELS  = IMG_CHANNELS + FLOW_CHANNELS   # 1 + 2 = 3
OUT_CHANNELS = IMG_CHANNELS                   # generator reconstructs the frame only

# ─────────────────────────────────────────────
#  DATASET SETTINGS
# ─────────────────────────────────────────────
# Supported datasets: "UCSD_Ped1" | "UCSD_Ped2" | "CUHK_Avenue"
DATASET_NAME    = "UCSD_Ped2"
TRAIN_SPLIT     = 0.9       # fraction of clips used for training
CLIP_LEN        = 16        # number of consecutive frames per clip (unused in frame-level mode)
FRAME_STRIDE    = 1         # sample every N-th frame from a video

# ─────────────────────────────────────────────
#  TRAINING HYPER-PARAMETERS
# ─────────────────────────────────────────────
BATCH_SIZE      = 16
NUM_EPOCHS      = 50
LR_G            = 2e-4      # generator learning rate
LR_D            = 2e-4      # discriminator learning rate
BETA1           = 0.5       # Adam β1 (standard GAN value)
BETA2           = 0.999     # Adam β2

# Loss weights (λ values)
LAMBDA_ADV      = 1.0       # adversarial loss weight
LAMBDA_RECON    = 10.0      # L1 reconstruction loss weight
LAMBDA_FLOW     = 5.0       # optical-flow consistency loss weight

# ─────────────────────────────────────────────
#  GENERATOR (U-Net)
# ─────────────────────────────────────────────
NGF             = 64        # base number of generator filters
NUM_DOWN_BLOCKS = 4         # encoder depth (also = decoder depth)

# ─────────────────────────────────────────────
#  DISCRIMINATOR (PatchGAN)
# ─────────────────────────────────────────────
NDF             = 64        # base number of discriminator filters
N_LAYERS_D      = 3         # number of Conv layers in PatchGAN

# ─────────────────────────────────────────────
#  EVALUATION / INFERENCE
# ─────────────────────────────────────────────
ANOMALY_THRESHOLD   = 0.5   # pixel-wise reconstruction MSE threshold (tunable)
TEMPORAL_SMOOTH_WINDOW = 15     # temporal smoothing window for anomaly scores (frames)
EVAL_BATCH_SIZE     = 1     # evaluate one frame at a time for score maps

# ─────────────────────────────────────────────
#  MISC
# ─────────────────────────────────────────────
DEVICE          = "cuda"    # "cuda" or "cpu"  (auto-fallback in train.py)
NUM_WORKERS     = 4         # DataLoader worker processes
SAVE_INTERVAL   = 5         # save checkpoint every N epochs
LOG_INTERVAL    = 1         # log training stats every N batches
SEED            = 42
