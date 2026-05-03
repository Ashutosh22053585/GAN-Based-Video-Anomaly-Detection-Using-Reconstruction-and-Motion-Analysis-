"""
generator.py
============
U-Net Generator for GAN-Based Video Anomaly Detection.

Input  : (B, IN_CHANNELS,  H, W) = (B, 3, 256, 256) [frame + u-flow + v-flow]
Output : (B, OUT_CHANNELS, H, W) = (B, 1, 256, 256) [reconstructed frame]

Skip connections allow the decoder to recover fine-grained spatial details lost
in the bottleneck. BatchNorm is applied everywhere except the first encoder block
and the output layer.
"""

import torch
import torch.nn as nn
import config


# ─────────────────────────────────────────────────────────────────────────────
#  BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    """Down-sampling block: Conv2d(stride=2) → [BatchNorm] → LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int,
                 use_bn: bool = True, slope: float = 0.2):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch,
                      kernel_size=4, stride=2, padding=1, bias=not use_bn)
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(slope, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """Up-sampling block: ConvTranspose2d(stride=2) → BatchNorm → ReLU → [Dropout]."""

    def __init__(self, in_ch: int, out_ch: int, dropout: bool = False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
#  U-NET GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class UNetGenerator(nn.Module):
    """
    U-Net Generator with skip connections.

    Channel progression (NGF=64):
        Encoder:    3 → 64 → 128 → 256 → 512
        Bottleneck: 512 → 512 (compress+expand, heavy dropout)
        Decoder:    1024 → 256 → 128 → 64
        Output:     64+64 → 1  (Tanh)
    """

    def __init__(self,
                 in_ch:  int = config.IN_CHANNELS,
                 out_ch: int = config.OUT_CHANNELS,
                 ngf:    int = config.NGF):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = EncoderBlock(in_ch,   ngf,     use_bn=False)  # 256→128
        self.enc2 = EncoderBlock(ngf,     ngf * 2)                # 128→64
        self.enc3 = EncoderBlock(ngf * 2, ngf * 4)                # 64→32
        self.enc4 = EncoderBlock(ngf * 4, ngf * 8)                # 32→16

        # ── Bottleneck ────────────────────────────────────────────────────────
        # Compress spatial map to 8×8 then expand back to 16×16
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        # ── Decoder (skip channels are concatenated before each block) ────────
        self.dec4 = DecoderBlock(ngf * 8 + ngf * 8, ngf * 4, dropout=True)   # 16→32
        self.dec3 = DecoderBlock(ngf * 4 + ngf * 4, ngf * 2, dropout=False)  # 32→64
        self.dec2 = DecoderBlock(ngf * 2 + ngf * 2, ngf,     dropout=False)  # 64→128

        # ── Output Layer ──────────────────────────────────────────────────────
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(ngf + ngf, out_ch,
                               kernel_size=4, stride=2, padding=1),   # 128→256
            nn.Tanh()
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        """DCGAN-style weight initialisation."""
        classname = m.__class__.__name__
        if classname in ("Conv2d", "ConvTranspose2d"):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif classname == "BatchNorm2d":
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) — [frame, flow_u, flow_v]
        Returns:
            (B, 1, H, W) reconstructed frame in [-1, 1]
        """
        e1 = self.enc1(x)                                 # (B,  64, 128, 128)
        e2 = self.enc2(e1)                                # (B, 128,  64,  64)
        e3 = self.enc3(e2)                                # (B, 256,  32,  32)
        e4 = self.enc4(e3)                                # (B, 512,  16,  16)

        b  = self.bottleneck(e4)                          # (B, 512,  16,  16)

        d4 = self.dec4(torch.cat([b,  e4], dim=1))       # (B, 256,  32,  32)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))       # (B, 128,  64,  64)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))       # (B,  64, 128, 128)

        out = self.output_layer(torch.cat([d2, e1], dim=1))  # (B, 1, 256, 256)
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  SMOKE-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = UNetGenerator()
    dummy = torch.randn(2, config.IN_CHANNELS, config.IMG_HEIGHT, config.IMG_WIDTH)
    out   = model(dummy)
    print(f"Input : {dummy.shape}  →  Output : {out.shape}")
    assert out.shape == (2, config.OUT_CHANNELS, config.IMG_HEIGHT, config.IMG_WIDTH)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")
    print("[Generator] Smoke test PASSED ✓")
