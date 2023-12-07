# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

"""
Task-guided Domain Gap Reduction for Monocular Depth Prediction in Endoscopy.

This repo is largely based on the code bases of "SharinGAN: Combining Synthetic and Real Data for Unsupervised GeometryEstimation"
(https://github.com/koutilya-pnvr/SharinGAN) and "EndoSLAM" (https://github.com/CapsuleEndoscope/EndoSLAM)

This software is licensed under the terms of a CC BY public copyright license.

Anita Rau, a.rau.16@ucl.ac.uk, 2023
"""

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0.2)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=1):
        super(SpatialAttention, self).__init__()
        kernel_size = 1
        padding = 3 if kernel_size == 7 else 0
        
        self.conv1 = nn.Conv2d(64, 4, kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(4, 4, kernel_size , padding=padding, bias=False)
        self.conv3 = nn.Conv2d(4, 64, kernel_size, padding=padding, bias=False)
        self.maxPooling = nn.MaxPool2d(4,stride=4)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.upsample = nn.Upsample(scale_factor=4)

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.maxPooling(x1)
        reshaped1 = torch.reshape(x2,(x2.shape[0],x2.shape[1],-1,x2.shape[2]))
        y = torch.matmul(reshaped1,x2)
        z = self.relu(y)
        z = self.conv2(z)
        t = self.softmax(z) 
        out1 = torch.matmul(t,reshaped1)
        conv3_out = self.conv3(out1)
        upsample_out = self.upsample(conv3_out)
        k = torch.reshape(upsample_out,(upsample_out.shape[0],upsample_out.shape[1],-1,upsample_out.shape[2]))
        output = k + x        
                
        return output


class ResnetEncoder2(nn.Module):
    """Pytorch module for a resnet encoder
    """
    # Adapted from Monodepth by EndoSLAM
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder2, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = input_image
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features
