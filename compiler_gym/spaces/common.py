# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from inspect import isclass
from numbers import Integral, Real

import numpy as np


def issubdtype(subtype, supertype):
    if isclass(subtype) and isclass(supertype) and issubclass(subtype, supertype):
        return True
    subdtype = np.dtype(subtype)
    superdtype = np.dtype(supertype)
    if np.dtype(subdtype) == np.dtype(superdtype):
        return True

    common_dtype = np.find_common_type([], [subdtype, superdtype])
    if not np.issubdtype(common_dtype, superdtype):
        return False
    if (
        issubclass(common_dtype.type, Real)
        and issubclass(subdtype.type, Integral)
        and 2 ** np.finfo(common_dtype).nmant < np.iinfo(subdtype).max
    ):
        return False
    return True
