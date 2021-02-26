# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np


def geometric_mean(iterable):
    vals = np.array(iterable)
    # Shortcut to return 0 when any element of the input is zero without
    # performaning (and printing a warning about) a division by zero below.
    if not np.all(vals):
        return 0
    a = np.log(vals)
    return np.exp(a.sum() / len(a))
