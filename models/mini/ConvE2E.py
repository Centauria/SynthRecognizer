# -*- coding: utf-8 -*-
import torch.nn as nn

import models.mini.Conv6XL


class ConvE2E(nn.Module):
    def __init__(self):
        super(ConvE2E, self).__init__()
        self.conv_1d = nn.Sequential(
            nn.Conv1d(2, 96, (64,), (4,)),
            nn.ReLU(),
            nn.Conv1d(96, 96, (32,), (4,)),
            nn.ReLU(),
            nn.Conv1d(96, 128, (16,), (4,)),
            nn.ReLU(),
            nn.Conv1d(128, 257, (8,), (4,)),
            nn.ReLU(),
        )
        self.conv_2d = models.mini.Conv6XL.model

    def forward(self, x):
        x = self.conv_1d(x)
        x = x[:, :, :64].transpose(1, 2).unsqueeze(1)
        x = self.conv_2d(x)
        return x


model = ConvE2E()
