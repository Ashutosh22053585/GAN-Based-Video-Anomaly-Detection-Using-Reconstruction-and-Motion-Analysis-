# GAN-Based Video Anomaly Detection: AI Workflow & Context

This document serves as a persistent context guide for AI assistants working on this project. It outlines the architecture, file structure, and standard workflows to ensure smooth onboarding and development in future sessions.

## 1. Project Objective
Build a robust Video Anomaly Detection system using Generative Adversarial Networks (GANs). The system learns the distribution of "normal" videos by predicting future frames and their optical flow. During inference, events that produce high prediction errors (both visually and dynamically) are flagged as anomalies.

## 2. Codebase Structure
- **`config.py`**: The central source of truth for hyperparameters, dataset paths, batch sizes, and learning rates. Always check this first when adjusting training behavior.
- **`dataloader.py`**: Custom PyTorch datasets and dataloaders. Responsible for loading consecutive video frames, applying transformations, and handling temporal sequences (e.g., $t$ past frames).
- **`generator.py`**: The predictive model (often U-Net based). Takes past frames as input and outputs the predicted next frame along with optical flow maps.
- **`discriminator.py`**: The adversarial model (typically PatchGAN). Evaluates whether a sequence of frames is "real" (ground truth) or "fake" (generated).
- **`loss.py`**: Contains the critical loss functions:
  - *Intensity Loss (L1)*: Pixel-wise difference.
  - *Gradient/Sharpness Loss*: Preserves edges.
  - *Flow Consistency Loss*: Ensures predicted motion aligns with actual motion.
  - *Adversarial Loss*: GAN min-max game loss.
- **`train.py`**: The main training loop. Handles optimization, logging, and checkpointing.
- **`evaluate.py`**: The inference pipeline. Calculates anomaly scores, applies temporal smoothing, and computes final metrics (like AUC-ROC).
- **`smoke_test.py`**: A vital utility script that runs dummy tensors through the models and loss functions. **Always run this after architectural changes to verify tensor dimensions.**
- **`PROJECT_EXPLAINED.md`**: Deep dive into the theory, mathematics, and analogies of the project.

## 3. Standard AI Action Workflow
When asked to modify or debug the project, follow this flow:
1. **Verify Config**: Ensure `config.py` settings align with the requested task (e.g., if changing image size, ensure it's updated in config).
2. **Component Update**: Make changes to the specific module (`generator.py`, `loss.py`, etc.).
3. **Smoke Test**: Run `python smoke_test.py` to guarantee no shape mismatches were introduced.
4. **Train/Eval**: Proceed to update `train.py` or `evaluate.py` if the integration logic changed.

## 4. Key AI Context Notes
- The project heavily relies on **PyTorch**.
- Avoid making arbitrary changes to tensor dimensions without tracing them through both the Generator and Discriminator.
- Anomaly scoring is highly sensitive to the weights assigned to the different losses in `loss.py`. When tuning detection accuracy, this is usually the primary focus area.
