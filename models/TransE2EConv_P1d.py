# -*- coding: utf-8 -*-
import torch

from models.ConvE2E import ConvE2E
from models.TransE2EConv import TransE2EConv


def get_model(conf: dict):
    model = TransE2EConv()
    e2e = ConvE2E()
    c = torch.load(conf['pretrained_model']['conv_1d'])
    e2e.load_state_dict(c['model'])
    model.conv_1d.load_state_dict(e2e.conv_1d.state_dict())
    for param in model.conv_1d.parameters():
        param.requires_grad = False
    return model
