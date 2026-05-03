"""
discriminator.py
================
PatchGAN Discriminator for GAN-Based Video Anomaly Detection.

PatchGAN judges whether overlapping N×N patches of an image are real or fake,
rather than evaluating the whole image as one scalar. This leads to sharper,
higher-frequency reconstructions and is the standard discriminator in pix2pix.

Input  : (B, IN_CHANNELS + OUT_CHANNELS, H, W) = (B, 4, 256, 256)
         The discriminator receives the concatenation of:
           - the generator's condition input  (3 ch: frame + flow)
           - the image being judged           (1 ch: real frame OR reconstructed frame)
Output : (B, 1, H_patch, W_patch)  — a patch-level real/fake score map
         For 256×256 input with N_LAYERS_D=3, the patch map is ~30×30.
"""

import torch
import torch.nn as nn
import config


# ─────────────────────────────────────────────────────────────────────────────
#  BUILDING BLOCK
# ─────────────────────────────────────────────────────────────────────────────

class PatchBlock(nn.Module):
    """
    A single PatchGAN conv block:
        Conv2d → [InstanceNorm] → LeakyReLU

    InstanceNorm (per-sample normalization) is preferred over BatchNorm in
    discriminators because it works well with small patch sizes and prevents
    per-batch mean leakage.
    """

    def __init__(self, in_ch: int, out_ch: int,
                 stride: int = 2, use_norm: bool = True):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch,
                      kernel_size=4, stride=stride,
                      padding=1, bias=not use_norm)
        ]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
#  PATCHGAN DISCRIMINATOR
# ─────────────────────────────────────────────────────────────────────────────

class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator.

    Builds N_LAYERS_D strided convolutional blocks, then a final conv that
    outputs a 2-D patch score map.  No sigmoid — use BCEWithLogitsLoss.

    Args:
        in_ch    : Channels of condition input (config.IN_CHANNELS = 3).
        target_ch: Channels of the target image (config.OUT_CHANNELS = 1).
        ndf      : Base filter count (config.NDF = 64).
        n_layers : Number of down-sampling conv layers (config.N_LAYERS_D = 3).
    """

    def __init__(self,
                 in_ch:     int = config.IN_CHANNELS,
                 target_ch: int = config.OUT_CHANNELS,
                 ndf:       int = config.NDF,
                 n_layers:  int = config.N_LAYERS_D):
        super().__init__()

        # Discriminator sees condition + target concatenated
        first_ch = in_ch + target_ch   # 3 + 1 = 4

        # ── Layer stack ───────────────────────────────────────────────────────
        blocks: list[nn.Module] = []

        # Layer 1: no normalization (standard PatchGAN practice)
        blocks.append(PatchBlock(first_ch, ndf, stride=2, use_norm=False))

        # Layers 2 … n_layers: channel doubles each step, capped at ndf*8
        curr_ch = ndf
        for layer_idx in range(1, n_layers):
            prev_ch  = curr_ch
            curr_ch  = min(curr_ch * 2, ndf * 8)
            # Last strided layer uses stride=1 to keep spatial resolution
            stride = 1 if layer_idx == n_layers - 1 else 2
            blocks.append(PatchBlock(prev_ch, curr_ch,
                                     stride=stride, use_norm=True))

        # Final 1-ch output (no activation — raw logits for BCEWithLogitsLoss)
        blocks.append(
            nn.Conv2d(curr_ch, 1, kernel_size=4, stride=1, padding=1)
        )

        self.model = nn.Sequential(*blocks)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        """DCGAN-style weight init."""
        classname = m.__class__.__name__
        if classname == "Conv2d":
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif classname == "InstanceNorm2d" and hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, condition: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            condition : (B, 3, H, W) — generator input [frame, flow_u, flow_v]
            target    : (B, 1, H, W) — real frame or generator's output

        Returns:
            patch_logits: (B, 1, H_p, W_p) — real/fake logit map
        """
        x = torch.cat([condition, target], dim=1)   # (B, 4, H, W)
        return self.model(x)


# ─────────────────────────────────────────────────────────────────────────────
#  SMOKE-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    disc = PatchGANDiscriminator()

    cond  = torch.randn(2, config.IN_CHANNELS,  config.IMG_HEIGHT, config.IMG_WIDTH)
    real  = torch.randn(2, config.OUT_CHANNELS, config.IMG_HEIGHT, config.IMG_WIDTH)
    patch = disc(cond, real)

    print(f"Condition : {cond.shape}")
    print(f"Target    : {real.shape}")
    print(f"Patch map : {patch.shape}")   # e.g. (2, 1, 30, 30)
    total = sum(p.numel() for p in disc.parameters())
    print(f"Parameters: {total:,}")
    print("[Discriminator] Smoke test PASSED ✓")
