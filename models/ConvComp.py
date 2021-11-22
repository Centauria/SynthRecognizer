# -*- coding: utf-8 -*-
import torch.nn as nn

from models.cnn import Conv2d


class ConvComp(nn.Module):
    def __init__(self):
        super(ConvComp, self).__init__()
        self.cnn = nn.Sequential(
            Conv2d(1, 64, (3, 3), (2, 2)),
            nn.ReLU(),
            Conv2d(64, 128, (3, 3), (2, 2)),
            nn.ReLU(),
            Conv2d(128, 128, (3, 4), (2, 3)),
            nn.ReLU(),
            Conv2d(128, 128, (3, 3), (2, 2)),
            nn.ReLU(),
            Conv2d(128, 256, (3, 3), (2, 2)),
            nn.ReLU(),
            Conv2d(256, 256, (3, 3), (1, 2)),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(5376, 512),
            nn.ReLU(),
        )
        self.fc_arg = nn.Sequential(
            nn.Linear(512, 10),
            nn.Sigmoid()
        )
        self.fc_classify = nn.Sequential(
            nn.Linear(512, 7)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        y = self.fc_arg(x)
        c = self.fc_classify(x)
        return dict(y_args=y, y_kind=c)


model = ConvComp()
