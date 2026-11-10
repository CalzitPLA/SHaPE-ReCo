#!/usr/bin/env python3
"""
Shared 3D-CNN voxel backbone used by all three task heads (type
classification, rotation regression, dimension regression).

Design notes (fixing the issues found in FeatureNet/ShaPe_Integra_v2.py's
trainer, whose training_history.json showed cls/reg loss as NaN from
epoch 1 with val_acc frozen at "always predict class 0"):

- No lazy-constructed FC layers sized from a runtime tensor shape (the
  old CNNRecognizer3D/GeometryParameterCNN pattern). Ends in
  AdaptiveAvgPool3d((1,1,1)) instead, so the backbone works at any input
  resolution and its output feature_dim is a compile-time constant -
  nothing depends on input size being exactly 64^3.
- Explicit Kaiming init on conv weights and default BatchNorm init
  (gamma=1, beta=0). NOT zero-init of the BN gamma: an earlier revision
  did that, but it is only valid for residual blocks (out = x + F(x)),
  and these blocks have no skip connection - it made every block's
  output identically zero, so the whole backbone returned a constant
  regardless of input. See _init_weights() below.
- No mixed-precision by default. The old trainer's NaNs coincided with
  torch.cuda.amp.autocast + GradScaler on an default-initialized
  residual stem; AMP can be re-enabled deliberately per-script once a
  fp32 baseline is confirmed stable.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, pool: bool):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


class Voxel3DBackbone(nn.Module):
    """Voxel grid (N,1,R,R,R) -> fixed-size feature vector (N, feature_dim)."""

    def __init__(self, in_channels: int = 1, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=7, stride=2, pool=False),
            ConvBlock(32, 64, kernel_size=5, stride=1, pool=True),
            ConvBlock(64, 96, kernel_size=3, stride=1, pool=True),
            ConvBlock(96, 128, kernel_size=3, stride=1, pool=True),
        )
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.proj = nn.Sequential(
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # NOT the "zero-init last BN gamma" trick from residual networks -
        # that only makes sense when a block's output is *added* to a skip
        # connection (out = x + F(x)), so zeroing F's last BN makes the
        # block start as an identity. ConvBlock has no skip connection
        # (forward is a plain pool(act(bn(conv(x)))) sequential transform),
        # so zeroing its BN gamma doesn't approximate identity - it makes
        # the block's output exactly zero for every input (gamma=0, beta=0
        # -> BN output is identically 0), which then stays zero through
        # every subsequent block (Conv3d has bias=False, so conv(0) = 0
        # too). The whole backbone collapsed to a constant zero feature
        # vector regardless of input, which is genuinely unrecoverable in
        # practice within a normal training budget (found via all three
        # CNNs' validation accuracy/error being frozen at chance level for
        # 50 epochs despite slowly decreasing loss - BatchNorm's own
        # gamma/beta still get gradients even at gamma=0, but nowhere near
        # fast enough against a decaying LR schedule to recover). Plain
        # Kaiming conv init + default BatchNorm init (gamma=1, beta=0) is
        # correct here.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.global_pool(x).flatten(1)
        return self.proj(x)
