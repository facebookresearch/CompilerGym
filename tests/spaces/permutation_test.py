# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pytest

from compiler_gym.spaces import Permutation, Scalar
from tests.test_main import main


def test_invalid_scalar_range_dtype():
    with pytest.raises(
        TypeError, match="Permutation space can have integral scalar range only."
    ):
        Permutation(name="", scalar_range=Scalar(name="", min=0, max=2, dtype=float))


def test_equal():
    assert Permutation(
        name="perm", scalar_range=Scalar(name="range", min=0, max=2, dtype=int)
    ) == Permutation(
        name="perm", scalar_range=Scalar(name="range", min=0, max=2, dtype=int)
    )


def test_not_equal():
    permutation = Permutation(
        name="perm", scalar_range=Scalar(name="range", min=0, max=2, dtype=int)
    )
    assert permutation != Permutation(
        name="perm", scalar_range=Scalar(name="range", min=0, max=1, dtype=int)
    )


if __name__ == "__main__":
    main()
