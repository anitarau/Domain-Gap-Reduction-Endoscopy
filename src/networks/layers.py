"""
Task-guided Domain Gap Reduction for Monocular Depth Prediction in Endoscopy.

This repo is largely based on the code bases of "SharinGAN: Combining Synthetic and Real Data for Unsupervised GeometryEstimation"
(https://github.com/koutilya-pnvr/SharinGAN) and "EndoSLAM" (https://github.com/CapsuleEndoscope/EndoSLAM)

This software is licensed under the terms of a CC BY public copyright license.

Anita Rau, a.rau.16@ucl.ac.uk, 2023
"""

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")