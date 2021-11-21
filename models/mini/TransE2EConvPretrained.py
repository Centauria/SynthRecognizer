# -*- coding: utf-8 -*-
import torch

from models.mini.ConvE2E import ConvE2E
from models.mini.TransE2EConv import TransE2EConv


def get_model(conf: dict):
    model = TransE2EConv()
    prev_model = ConvE2E()
    checkpoint = torch.load(conf['pretrained_model']['conv_1d'])
    prev_model.load_state_dict(checkpoint['model'])
    model.conv_1d.load_state_dict(prev_model.conv_1d.state_dict())
    for param in model.conv_1d.parameters():
        param.requires_grad = False
    return model
