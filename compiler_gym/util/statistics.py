# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np


def geometric_mean(iterable):
    vals = np.array(iterable)
    if not len(vals):
        return 0
    # Shortcut to return 0 when any element of the input is not positive.
    if not np.all(vals > 0):
        return 0
    a = np.log(vals)
    return np.exp(a.sum() / len(a))
