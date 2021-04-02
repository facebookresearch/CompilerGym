# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np


def geometric_mean(iterable):
    """Zero-length-safe geometric mean."""
    values = np.asarray(iterable)
    if not values.size:
        return 0
    # Shortcut to return 0 when any element of the input is not positive.
    if not np.all(values > 0):
        return 0
    a = np.log(values)
    return np.exp(a.sum() / len(a))


def arithmetic_mean(iterable):
    """Zero-length-safe arithmetic mean."""
    values = np.asarray(iterable)
    if not values.size:
        return 0
    return values.mean()


def stdev(iterable):
    """Zero-length-safe standard deviation."""
    values = np.asarray(iterable)
    if not values.size:
        return 0
    return values.std()
