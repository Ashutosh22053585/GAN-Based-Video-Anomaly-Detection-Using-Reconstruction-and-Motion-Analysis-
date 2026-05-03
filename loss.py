"""
loss.py
=======
Combined loss functions for GAN-Based Video Anomaly Detection.

Total Generator Loss = λ_adv  * L_adversarial
                     + λ_recon * L_reconstruction  (L1)
                     + λ_flow  * L_flow_consistency (L1 on flow channels)

Adversarial loss: standard least-squares GAN (LSGAN) — more stable than
vanilla binary cross-entropy because it does not saturate.

Reconstruction loss: pixel-wise L1 between the generated and real frame.
L1 is preferred over L2 because it produces sharper edges.

Flow consistency loss: forces the generator to produce frames whose implied
optical flow (taken from the input flow channels) is consistent with the
reconstructed content, penalising motion artefacts.
"""

import torch
import torch.nn as nn
import config


# ─────────────────────────────────────────────────────────────────────────────
#  LSGAN ADVERSARIAL LOSS
# ─────────────────────────────────────────────────────────────────────────────

class LSGANLoss(nn.Module):
    """
    Least-Squares GAN loss (Mao et al., 2017).

    For the discriminator:
        L_D = 0.5 * E[(D(real) - 1)^2] + 0.5 * E[(D(fake))^2]

    For the generator:
        L_G_adv = 0.5 * E[(D(fake) - 1)^2]

    Args:
        target_real_label : label value for real patches (default 1.0)
        target_fake_label : label value for fake patches (default 0.0)
    """

    def __init__(self,
                 target_real_label: float = 1.0,
                 target_fake_label: float = 0.0):
        super().__init__()
        self.register_buffer("real_label",
                             torch.tensor(target_real_label))
        self.register_buffer("fake_label",
                             torch.tensor(target_fake_label))
        self.criterion = nn.MSELoss()

    def _expand_label(self, prediction: torch.Tensor,
                      target_is_real: bool) -> torch.Tensor:
        """Broadcast scalar label to match prediction shape."""
        label = self.real_label if target_is_real else self.fake_label
        return label.expand_as(prediction)

    def discriminator_loss(self,
                           pred_real: torch.Tensor,
                           pred_fake: torch.Tensor) -> torch.Tensor:
        """
        Full discriminator LSGAN loss.

        Args:
            pred_real: D output on real  data — (B, 1, H_p, W_p)
            pred_fake: D output on fake  data — (B, 1, H_p, W_p)
        Returns:
            Scalar discriminator loss.
        """
        loss_real = self.criterion(pred_real, self._expand_label(pred_real, True))
        loss_fake = self.criterion(pred_fake, self._expand_label(pred_fake, False))
        return 0.5 * (loss_real + loss_fake)

    def generator_loss(self, pred_fake: torch.Tensor) -> torch.Tensor:
        """
        Generator adversarial loss: fool the discriminator.

        Args:
            pred_fake: D output on generated frames — (B, 1, H_p, W_p)
        Returns:
            Scalar adversarial loss.
        """
        return self.criterion(pred_fake, self._expand_label(pred_fake, True))

    def forward(self, prediction: torch.Tensor,
                target_is_real: bool) -> torch.Tensor:
        """Generic forward: compute loss for given prediction and label side."""
        target = self._expand_label(prediction, target_is_real)
        return self.criterion(prediction, target)


# ─────────────────────────────────────────────────────────────────────────────
#  RECONSTRUCTION LOSS  (L1)
# ─────────────────────────────────────────────────────────────────────────────

class ReconstructionLoss(nn.Module):
    """
    Pixel-wise L1 loss between the generated and real frame.

    L_recon = mean |G(x) - y_real|

    L1 is chosen over L2 because it is less likely to produce blurry results
    (L2 penalises large errors quadratically, encouraging over-smoothing).
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, generated: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            generated : (B, 1, H, W)  generator output in [-1, 1]
            target    : (B, 1, H, W)  real frame in [-1, 1]
        Returns:
            Scalar L1 loss.
        """
        return self.criterion(generated, target)


# ─────────────────────────────────────────────────────────────────────────────
#  OPTICAL FLOW CONSISTENCY LOSS
# ─────────────────────────────────────────────────────────────────────────────

# class FlowConsistencyLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.l1 = nn.L1Loss()
#         sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32).unsqueeze(0)
#         sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32).unsqueeze(0)
#         self.register_buffer("sobel_x", sobel_x)
#         self.register_buffer("sobel_y", sobel_y)
# 
#     def _image_gradients(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         grad_x = nn.functional.conv2d(img, self.sobel_x, padding=1)
#         grad_y = nn.functional.conv2d(img, self.sobel_y, padding=1)
#         return grad_x, grad_y
# 
#     def forward(self, input_tensor: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
#         flow_u = input_tensor[:, 1:2, :, :]
#         flow_v = input_tensor[:, 2:3, :, :]
#         recon_u, recon_v = self._image_gradients(generated)
#         recon_u = torch.clamp(recon_u / 4.0, -1.0, 1.0)
#         recon_v = torch.clamp(recon_v / 4.0, -1.0, 1.0)
#         loss_u = self.l1(recon_u, flow_u)
#         loss_v = self.l1(recon_v, flow_v)
#         return (loss_u + loss_v) * 0.5

class WarpingFlowLoss(nn.Module):
    """
    Computes flow consistency by warping the previous frame using the input flow 
    and comparing it to the generated frame via L1 loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, prev_frame: torch.Tensor, flow_uv: torch.Tensor, generated_frame: torch.Tensor) -> torch.Tensor:
        B, C, H, W = prev_frame.shape
        
        # Build a sampling grid from flow using torch.meshgrid
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=prev_frame.device),
            torch.linspace(-1.0, 1.0, W, device=prev_frame.device),
            indexing='ij'
        )
        # grid shape: (B, H, W, 2)
        grid = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        
        # flow_uv is (B, 2, H, W), we need to permute to (B, H, W, 2)
        flow_offsets = flow_uv.permute(0, 2, 3, 1)
        
        # Warp prev_frame
        warp_grid = grid + flow_offsets
        warped_prev = torch.nn.functional.grid_sample(prev_frame, warp_grid, mode='bilinear', align_corners=True)
        
        return torch.nn.functional.l1_loss(warped_prev, generated_frame)

# ─────────────────────────────────────────────────────────────────────────────
#  COMBINED LOSS WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyGANLoss:
    """
    Convenience wrapper that holds all loss modules and computes the
    total generator and discriminator losses in one call.

    Usage:
        criterion = AnomalyGANLoss(device)
        loss_dict = criterion.generator_total(input_t, generated, target, pred_fake)
        loss_D    = criterion.discriminator_total(pred_real, pred_fake)
    """

    def __init__(self, device: torch.device):
        self.adv_criterion   = LSGANLoss().to(device)
        self.recon_criterion = ReconstructionLoss().to(device)
        self.flow_criterion  = WarpingFlowLoss().to(device)

        self.lambda_adv   = config.LAMBDA_ADV
        self.lambda_recon = config.LAMBDA_RECON
        self.lambda_flow  = config.LAMBDA_FLOW

    def generator_total(self,
                        input_tensor: torch.Tensor,
                        prev_frame:   torch.Tensor,
                        generated:    torch.Tensor,
                        target:       torch.Tensor,
                        pred_fake:    torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute all generator losses and return as a dict.

        Returns:
            {
              'adv':   adversarial loss (scalar),
              'recon': reconstruction  loss (scalar),
              'flow':  flow consistency loss (scalar),
              'total': weighted sum (scalar),
            }
        """
        l_adv   = self.adv_criterion.generator_loss(pred_fake)
        l_recon = self.recon_criterion(generated, target)
        
        flow_uv = input_tensor[:, 1:3, :, :]
        l_flow  = self.flow_criterion(prev_frame, flow_uv, generated)

        l_total = (self.lambda_adv   * l_adv
                 + self.lambda_recon * l_recon
                 + self.lambda_flow  * l_flow)

        return {"adv": l_adv, "recon": l_recon, "flow": l_flow, "total": l_total}

    def discriminator_total(self,
                            pred_real: torch.Tensor,
                            pred_fake: torch.Tensor) -> torch.Tensor:
        """
        Compute discriminator LSGAN loss.

        Args:
            pred_real: D(condition, real_frame)  — (B, 1, H_p, W_p)
            pred_fake: D(condition, generated)   — (B, 1, H_p, W_p)
        Returns:
            Scalar discriminator loss.
        """
        return self.adv_criterion.discriminator_loss(pred_real, pred_fake)


# ─────────────────────────────────────────────────────────────────────────────
#  SMOKE-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device    = torch.device("cpu")
    criterion = AnomalyGANLoss(device)

    B, H, W = 2, config.IMG_HEIGHT, config.IMG_WIDTH
    inp   = torch.randn(B, config.IN_CHANNELS,  H, W)
    gen   = torch.randn(B, config.OUT_CHANNELS, H, W)
    tgt   = torch.randn(B, config.OUT_CHANNELS, H, W)
    fake  = torch.randn(B, 1, 30, 30)   # simulated patch map
    real_ = torch.randn(B, 1, 30, 30)

    g_losses = criterion.generator_total(inp, gen, tgt, fake)
    d_loss   = criterion.discriminator_total(real_, fake)

    print("Generator losses:")
    for k, v in g_losses.items():
        print(f"  {k}: {v.item():.4f}")
    print(f"Discriminator loss: {d_loss.item():.4f}")
    print("[Loss] Smoke test PASSED ✓")
