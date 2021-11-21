# -*- coding: utf-8 -*-
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(257, 512),
            nn.ReLU()
        )
        self.transformer = nn.Transformer(d_model=512, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(512, 8),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.linear(x).squeeze(1)
        x = self.transformer(x, x).unsqueeze(1)
        x = self.out(x)
        return x


model = Transformer()
