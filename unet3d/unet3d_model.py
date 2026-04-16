import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConcatResidualBlock3D(nn.Module):
    """
    A 3D Residual Block that uses Concatenation instead of Addition.
    Maintains feature maps from early layers to prevent signal wash-out.
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        # Processing Branch
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, padding_mode='reflect')
        self.bn1 = nn.InstanceNorm3d(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1, padding_mode='reflect')
        self.bn2 = nn.InstanceNorm3d(out_c)

        # Identity/Residual Branch
        # If in_c != out_c, we use a 1x1 conv to match the depth
        self.identity_conv = nn.Conv3d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()

        # Bottleneck: Compresses the [out_c + out_c] concatenated features back to out_c
        self.bottleneck = nn.Conv3d(out_c * 2, out_c, kernel_size=1)

    def forward(self, x):
        identity = self.identity_conv(x)

        # Nonlinear processing
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Concatenate Processing + Identity
        combined = torch.cat([out, identity], dim=1)

        # Compress and Activate
        return F.relu(self.bottleneck(combined))


class ConcatDeepResUNet3D(nn.Module):
    def __init__(self, base_channels=16):
        super().__init__()

        # Encoder (Downsampling)
        self.enc1 = ConcatResidualBlock3D(1, base_channels)
        self.enc2 = ConcatResidualBlock3D(base_channels, base_channels * 2)
        self.enc3 = ConcatResidualBlock3D(base_channels * 2, base_channels * 4)

        # Bottleneck
        self.bottleneck = ConcatResidualBlock3D(base_channels * 4, base_channels * 8)

        # Decoder (Upsampling)
        # We use ConvTranspose3d to double spatial resolution
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConcatResidualBlock3D(base_channels * 8, base_channels * 4) # 8 = up4 + skip4

        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConcatResidualBlock3D(base_channels * 4, base_channels * 2) # 4 = up2 + skip2

        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConcatResidualBlock3D(base_channels * 2, base_channels) # 2 = up1 + skip1

        # Final Output (Logits for BCEWithLogitsLoss)
        self.final = nn.Conv3d(base_channels, 1, kernel_size=1)

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        p1 = F.max_pool3d(e1, 2)

        e2 = self.enc2(p1)
        p2 = F.max_pool3d(e2, 2)

        e3 = self.enc3(p2)
        p3 = F.max_pool3d(e3, 2)


        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder path with Skip Connections (Concatenation)
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final(d1)

