from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from .van import VAN, Block, SuperResBlock
from .hr_layers import ConvBlock, fSEModule, Conv3x3, Conv1x1, upsample, ConvBlockClassic

class DepthDecoder(nn.Module):
    """
    Monodepth2 decoder from the paper Digging Into Self-Supervised Monocular Depth Estimation
    Source code from https://github.com/nianticlabs/monodepth2
    """
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs


class HRDepthDecoder(nn.Module):
    """
    Backbone adopted from paper HR-Depth:
    https://github.com/shawLyu/HR-Depth/blob/main/networks/HR_Depth_Decoder.py
    """
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_super_res=True, convnext=False):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.num_ch_enc = num_ch_enc
        self.scales = scales
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.all_position = ["01", "11", "21", "31", "02", "12", "22", "03", "13", "04"]
        self.attention_position = ["31", "22", "13", "04"]
        self.non_attention_position = ["01", "11", "21", "02", "12", "03"]

        self.convs = nn.ModuleDict()
        self.van_blocks = nn.ModuleDict()
        self.upscaler = SuperResBlock(16) if use_super_res else upsample

        for j in range(5):
            for i in range(5 - j):
                # upconv 0
                num_ch_in = num_ch_enc[i]
                if i == 0 and j != 0:
                    num_ch_in /= 2
                num_ch_out = num_ch_in / 2
                self.convs["X_{}{}_Conv_0".format(i, j)] = ConvBlock(num_ch_in, num_ch_out, convnext)

                # X_04 upconv 1, only add X_04 convolution
                if i == 0 and j == 4:
                    num_ch_in = num_ch_out
                    num_ch_out = self.num_ch_dec[i]
                    self.convs["X_{}{}_Conv_1".format(i, j)] = ConvBlock(num_ch_in, num_ch_out, convnext)

        # declare fSEModule and original module
        for index in self.attention_position:
            row = int(index[0])
            col = int(index[1])

            fse_high_ch = num_ch_enc[row + 1] // 2
            fse_low_ch = self.num_ch_enc[row] + self.num_ch_dec[row + 1] * (col - 1)
            fse_out_ch = fse_high_ch if index == "04" else self.num_ch_enc[row]

            self.convs["X_" + index + "_attention"] = fSEModule(fse_high_ch, fse_low_ch, fse_out_ch, use_super_res)

        for index in self.non_attention_position:
            row = int(index[0])
            col = int(index[1])

            if col == 1:
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(num_ch_enc[row + 1] // 2 + self.num_ch_enc[row],
                                                                                 self.num_ch_dec[row + 1],
                                                                                 convnext)
            else:
                self.convs["X_"+index+"_downsample"] = Conv1x1(num_ch_enc[row+1] // 2 + self.num_ch_enc[row]
                                                               + self.num_ch_dec[row+1]*(col-1), self.num_ch_dec[row + 1] * 2)
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(self.num_ch_dec[row + 1] * 2,
                                                                                 self.num_ch_dec[row + 1], convnext)

        for i in range(4):
            self.convs["dispConvScale{}".format(i)] = ConvBlock(self.num_ch_dec[i], self.num_output_channels, convnext=True)

        self.sigmoid = nn.Sigmoid()

    def nestConv(self, conv, high_feature, low_features):
        conv_0 = conv[0]
        conv_1 = conv[1]
        assert isinstance(low_features, list)
        high_features = [upsample(conv_0(high_feature))]
        for feature in low_features:
            high_features.append(feature)
        high_features = torch.cat(high_features, 1)
        if len(conv) == 3:
            high_features = conv[2](high_features)
        return conv_1(high_features)

    def forward(self, input_features):
        outputs = {}
        features = {}
        for i in range(5):
            features["X_{}0".format(i)] = input_features[i]
        # Network architecture
        for index in self.all_position:
            row = int(index[0])
            col = int(index[1])

            low_features = []
            for i in range(col):
                low_features.append(features["X_{}{}".format(row, i)])

            # add fSE block to decoder
            if index in self.attention_position:
                features["X_"+index] = self.convs["X_" + index + "_attention"](
                    self.convs["X_{}{}_Conv_0".format(row+1, col-1)](features["X_{}{}".format(row+1, col-1)]), low_features)
            elif index in self.non_attention_position:
                conv = [self.convs["X_{}{}_Conv_0".format(row + 1, col - 1)],
                        self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)]]
                if col != 1:
                    conv.append(self.convs["X_" + index + "_downsample"])
                features["X_" + index] = self.nestConv(conv, features["X_{}{}".format(row+1, col-1)], low_features)

        x = features["X_04"]
        x = self.convs["X_04_Conv_0"](x)
        x = self.convs["X_04_Conv_1"](self.upscaler(x))  # shape before upsample - (16, H / 2, W / 2)
        outputs[('disp', 0)] = self.sigmoid(self.convs["dispConvScale0"](x))  # (16, 1)
        outputs[('disp', 1)] = self.sigmoid(self.convs["dispConvScale1"](features["X_04"]))  # (32, 1)
        outputs[('disp', 2)] = self.sigmoid(self.convs["dispConvScale2"](features["X_13"]))  # (64, 1)
        outputs[('disp', 3)] = self.sigmoid(self.convs["dispConvScale3"](features["X_22"]))  # (128, 1)
        return outputs