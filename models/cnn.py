# -*- coding: utf-8 -*-
import functools
import operator

import torch.nn as nn


class Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(*args, **kwargs)
        pad_size = functools.reduce(
            operator.__add__,
            [[d * (k // 2), d * ((k - 1) // 2)] for k, d in
             zip(self.kernel_size[::-1],
                 self.dilation[::-1])]
        )
        self.zero_pad_2d = nn.ZeroPad2d(pad_size)

    def forward(self, x):
        return self._conv_forward(self.zero_pad_2d(x), self.weight, self.bias)


class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(*args, **kwargs)
        pad_size = functools.reduce(
            operator.__add__,
            [[d * (k // 2), d * ((k - 1) // 2)] for k, d in
             zip(self.kernel_size[::-1],
                 self.dilation[::-1])]
        )
        self.zero_pad_2d = nn.ZeroPad2d(pad_size)

    def forward(self, x):
        return self._conv_forward(self.zero_pad_2d(x), self.weight, self.bias)
