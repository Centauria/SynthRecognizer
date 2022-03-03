# -*- coding: utf-8 -*-
import torch.nn as nn

from models.ConvComp import ConvComp
from models.conformer import ConformerEncoder


class ConfE2EConv(nn.Module):
    def __init__(self):
        super(ConfE2EConv, self).__init__()
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
        self.transformer = nn.Transformer(d_model=256,
                                          custom_encoder=ConformerEncoder(input_dim=256,
                                                                          num_heads=8,
                                                                          ffn_dim=2048,
                                                                          num_layers=6,
                                                                          depthwise_conv_kernel_size=31,
                                                                          dropout=0.1),
                                          batch_first=True)
        self.conv_comp = ConvComp()

    def forward(self, x):
        x, s = x
        x = self.conv_1d(x)
        x = x.transpose(1, 2)
        x = self.ins(x)
        s = self.ins(s.squeeze(1))
        x = self.transformer(x, s).unsqueeze(1)
        x = self.conv_comp(x)
        return x


model = ConfE2EConv()
