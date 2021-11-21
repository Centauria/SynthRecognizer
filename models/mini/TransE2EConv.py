# -*- coding: utf-8 -*-
import torch.nn as nn

from models.cnn import Conv2d


class TransE2EConv(nn.Module):
    def __init__(self):
        super(TransE2EConv, self).__init__()
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
        self.ins = nn.Sequential(
            nn.Linear(257, 256),
            nn.ReLU()
        )
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
            nn.ReLU()
        )
        self.transformer = nn.Transformer(d_model=256, batch_first=True)
        self.out = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x, s = x
        x = self.conv_1d(x)
        x = x[:, :, :64].transpose(1, 2)
        x = self.ins(x)
        s = self.ins(s.squeeze(1))
        x = self.transformer(x, s).unsqueeze(1)
        x = self.cnn(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.transformer(x, x)
        x = self.out(x)
        return x


model = TransE2EConv()
