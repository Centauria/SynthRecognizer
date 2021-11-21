# -*- coding: utf-8 -*-
import torch.nn as nn

from models.cnn import Conv2d

model = nn.Sequential(
    Conv2d(1, 64, (3, 3), (2, 2)),
    nn.LayerNorm([64, 32, 129]),
    nn.ReLU(),
    Conv2d(64, 128, (3, 3), (2, 2)),
    nn.LayerNorm([128, 16, 65]),
    nn.ReLU(),
    Conv2d(128, 128, (3, 4), (2, 3)),
    nn.LayerNorm([128, 8, 22]),
    nn.ReLU(),
    Conv2d(128, 128, (3, 3), (2, 2)),
    nn.LayerNorm([128, 4, 11]),
    nn.ReLU(),
    Conv2d(128, 256, (3, 3), (2, 2)),
    nn.LayerNorm([256, 2, 6]),
    nn.ReLU(),
    Conv2d(256, 256, (3, 3), (1, 2)),
    nn.LayerNorm([256, 2, 3]),
    nn.ReLU(),
    nn.Flatten(1),
    nn.Linear(1536, 512),
    nn.ReLU(),
    nn.Linear(512, 4),
    nn.Sigmoid()
)
