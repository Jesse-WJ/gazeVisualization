#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021-09-22 11:20
# @Author  : yunshang
# @FileName: alexnet.py
# @Software: PyCharm
# @desc    : 

import torch
import torchvision
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_output=2):
        super(AlexNet, self).__init__()
        # 36 * 60
        # 18 * 30
        # 9  * 15

        alexnet = torchvision.models.alexnet(pretrained=True)

        self.convNet = alexnet.features

        self.weightStream = nn.Sequential(
            nn.Conv2d(256, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
        )

        self.FC = nn.Sequential(
            nn.Linear(256 * 13 * 13, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_output)
        )

    def forward(self, x_in):
        faceFeature = self.convNet(x_in)
        weight = self.weightStream(faceFeature)

        faceFeature = weight * faceFeature

        faceFeature = torch.flatten(faceFeature, start_dim=1)
        gaze = self.FC(faceFeature)

        return gaze

def get_model(num_output=2):
    model = AlexNet(num_output=num_output)
    return model


if __name__ == "__main__":
    # model = AlexNet(num_output=2)
    model = get_model(num_output=2)
    x = {"face":torch.rand((4, 3, 224, 224))}
    print(model)
    print(model(x['face']))
