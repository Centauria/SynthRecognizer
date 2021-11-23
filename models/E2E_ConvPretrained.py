# -*- coding: utf-8 -*-
import torch

from models.ConvE2E import ConvE2E


def get_model(conf: dict):
    model = ConvE2E()
    checkpoint = torch.load(conf['pretrained_model']['conv_2d'])
    model.conv_2d.load_state_dict(checkpoint['model'])
    for param in model.conv_2d.parameters():
        param.requires_grad = False
    return model
