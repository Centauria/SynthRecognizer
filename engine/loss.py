# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class LogCoshLoss(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps
    
    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + self.eps)))


def mse():
    return nn.MSELoss()


def lcl():
    return LogCoshLoss()


def bce():
    return nn.BCELoss()


def ce():
    return nn.CrossEntropyLoss()
