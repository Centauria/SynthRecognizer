# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class ConvolutionalRNN(nn.Module):
    def __init__(self):
        super(ConvolutionalRNN, self).__init__()
