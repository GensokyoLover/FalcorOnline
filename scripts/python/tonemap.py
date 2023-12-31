import numpy as np
import torch


def log1p_tensor(x):
    y = torch.log1p(x)
    return y


def inv_log1p_tensor(y):
    x = torch.exp(y) - 1
    return x


def log1p(x):
    y = np.log1p(x)
    return y


def inv_log1p(y):
    x = np.exp(y) - 1
    return x


def n2n(x):
    y = (x / (1 + x))
    return y


def inv_n2n(y):
    y = torch.clip(y, 0, 0.99)
    x = y / (1 - y)
    return x
