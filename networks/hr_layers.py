from __future__ import absolute_import, division, print_function

import numpy as np

# from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .van import SuperResBlock, VAN_Block
from .van import AttentionModule as LKA


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, convnext: bool):
        super().__init__()
        in_channels, out_channels = int(in_channels), int(out_channels)
        if convnext:
            self.convblock = ConvNeXtBlock(in_channels, out_channels)
        else:
            self.convblock = ConvBlockClassic(in_channels, out_channels)

    def forward(self, x):
        return self.convblock.forward(x)


class ConvBlockClassic(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, out_dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depth-wise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 2 * dim)  # point-wise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(2 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.head = nn.Conv2d(dim, out_dim, kernel_size=1)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + x
        x = self.head(x)
        return x


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super().__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class fSEModule(nn.Module):
    def __init__(self, high_feature_channel, low_feature_channels, output_channel=None, use_super_res=True, use_ca=True):
        super().__init__()
        in_channel = high_feature_channel + low_feature_channels
        out_channel = high_feature_channel
        if output_channel is not None:
            out_channel = output_channel
        channel = in_channel
        self.upscaler = SuperResBlock(high_feature_channel) if use_super_res else upsample
        self.use_channel_attention = use_ca
        if use_ca:
            reduction = 16
            self.lka = LKA(channel)
            self.lka_conv_1 = nn.Conv2d(channel, channel, 1)
            self.lka_conv_2 = nn.Conv2d(channel, channel, 1)
            self.lka_activation = nn.Mish()

            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel, bias=False)
            )
            self.sigmoid = nn.Sigmoid()
            self.conv_se = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1)
            self.final_activation = nn.Mish()
        else:
            self.norm = nn.BatchNorm2d(channel)
            self.lka = LKA(channel)
            self.lka_conv_1 = nn.Conv2d(channel, channel, 1)
            self.lka_conv_2 = nn.Conv2d(channel, out_channel, 1)
            self.lka_activation = nn.Mish()

    def forward(self, high_features, low_features):
        features = [self.upscaler(high_features)]
        features += low_features
        features = torch.cat(features, 1)

        if self.use_channel_attention:
            b, c, _, _ = features.size()
            features = self.lka_conv_2(self.lka(self.lka_activation(self.lka_conv_1(features)))) + features
            y = self.avg_pool(features).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            y = self.sigmoid(y)
            features = features * y.expand_as(features)
            features = self.final_activation(self.conv_se(features))
        else:
            features = self.lka(self.lka_activation(self.lka_conv_1(self.norm(features)))) + features
            features = self.lka_conv_2(features)
        return features


class ImageLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x
