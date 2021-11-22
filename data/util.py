# -*- coding: utf-8 -*-
import torch.utils.data


def random_split(dataset, ratio):
    size = len(dataset)
    s = sum(ratio)
    r = [x / s for x in ratio]
    n = [int(size * x) for x in r]
    n[-1] = size - sum(n[:-1])
    return torch.utils.data.random_split(dataset, n)
