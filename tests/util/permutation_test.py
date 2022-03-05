# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np

import compiler_gym.util.permutation as permutation
from tests.test_main import main


def test_permutation_number_mapping():
    original_permutation = np.array([4, 3, 1, 5, 2, 6, 0], dtype=int)
    permutation_number = permutation.convert_permutation_to_number(original_permutation)
    mapped_permutation = permutation.convert_number_to_permutation(
        n=permutation_number, permutation_size=len(original_permutation)
    )
    assert np.array_equal(original_permutation, mapped_permutation)

    original_permutation2 = np.array([2, 0, 5, 1, 4, 6, 3], dtype=int)
    permutation_number2 = permutation.convert_permutation_to_number(
        original_permutation2
    )
    mapped_permutation2 = permutation.convert_number_to_permutation(
        n=permutation_number2, permutation_size=len(original_permutation2)
    )
    assert np.array_equal(original_permutation2, mapped_permutation2)


if __name__ == "__main__":
    main()
