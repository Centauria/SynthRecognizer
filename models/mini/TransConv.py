# -*- coding: utf-8 -*-
import torch.nn as nn

import models.mini.Conv6XL


class TransConv(nn.Module):
    def __init__(self):
        super(TransConv, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(257, 512),
            nn.ReLU()
        )
        self.transformer = nn.Transformer(d_model=512, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(512, 257),
            nn.ReLU(),
            models.mini.Conv6XL.model
        )
    
    def forward(self, x):
        x = self.linear(x).squeeze(1)
        x = self.transformer(x, x).unsqueeze(1)
        x = self.out(x)
        return x


model = TransConv()
