# -*- coding: utf-8 -*-
import torch.nn as nn
from ignite.metrics import RootMeanSquaredError, MeanSquaredError, MeanAbsoluteError, Loss

from engine import loss


def rmse(output_transform=lambda x, y, y_pred: (y_pred, y)):
    return RootMeanSquaredError(output_transform)


def lcl(output_transform=lambda x, y, y_pred: (y_pred, y)):
    return Loss(loss.LogCoshLoss(), output_transform)


def mse(output_transform=lambda x, y, y_pred: (y_pred, y)):
    return MeanSquaredError(output_transform)


def mae(output_transform=lambda x, y, y_pred: (y_pred, y)):
    return MeanAbsoluteError(output_transform)


def bce(output_transform=lambda x, y, y_pred: (y_pred, y)):
    return Loss(nn.BCELoss(), output_transform)


def ce(output_transform=lambda x, y, y_pred: (y_pred, y)):
    return Loss(nn.CrossEntropyLoss(), output_transform)
