# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from numbers import Integral
from typing import List

import numpy as np


def convert_number_to_permutation(
    n: Integral, permutation_size: Integral
) -> List[Integral]:
    m = n
    res = np.zeros(permutation_size, dtype=type(permutation_size))
    elements = np.arange(permutation_size, dtype=type(permutation_size))
    for i in range(permutation_size):
        j = m % (permutation_size - i)
        m = m // (permutation_size - i)
        res[i] = elements[j]
        elements[j] = elements[permutation_size - i - 1]
    return res


def convert_permutation_to_number(permutation: List[Integral]) -> Integral:
    pos = np.arange(len(permutation), dtype=int)
    elements = np.arange(len(permutation), dtype=int)
    m = 1
    res = 0
    for i in range(len(permutation) - 1):
        res += m * pos[permutation[i]]
        m = m * (len(permutation) - i)
        pos[elements[len(permutation) - i - 1]] = pos[permutation[i]]
        elements[pos[permutation[i]]] = elements[len(permutation) - i - 1]
    return res
