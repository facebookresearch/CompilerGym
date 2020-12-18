# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np


def geometric_mean(iterable):
    a = np.log(iterable)
    return np.exp(a.sum() / len(a))
