import numpy as np
import torch


def rescale(x, eps=1e-3):
    sign = get_sign(x)
    x_abs = get_abs(x)
    if isinstance(x, np.ndarray):
        return sign * (np.sqrt(x_abs + 1) - 1) + eps * x
    else:
        return sign * ((x_abs + 1).sqrt() - 1) + eps * x


def inv_rescale(x, eps=1e-3):
    sign = get_sign(x)
    x_abs = get_abs(x)
    if eps == 0:
        return sign * (x * x + 2.0 * x_abs)
    else:
        return sign * (
            (((1.0 + 4.0 * eps * (x_abs + 1.0 + eps)).sqrt() - 1.0) / (2.0 * eps)).pow(
                2
            )
            - 1.0
        )


def get_sign(x):
    if isinstance(x, np.ndarray):
        return np.sign(x)
    elif isinstance(x, torch.Tensor):
        return x.sign()
    else:
        raise NotImplementedError(f"Data type: {type(x)} is not implemented")


def get_abs(x):
    if isinstance(x, np.ndarray):
        return np.abs(x)
    elif isinstance(x, torch.Tensor):
        return x.abs()
    else:
        raise NotImplementedError(f"Data type: {type(x)} is not implemented")
