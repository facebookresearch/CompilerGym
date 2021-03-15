# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/util:statistics."""
from pytest import approx

from compiler_gym.util.statistics import geometric_mean
from tests.test_main import main


def test_geometric_mean_empty_list():
    assert geometric_mean([]) == 0


def test_geometric_mean_zero_value():
    assert geometric_mean([0, 1, 2]) == 0


def test_geometric_mean_negative():
    assert geometric_mean([-1, 1, 2]) == 0


def test_geometric_mean_123():
    assert geometric_mean([1, 2, 3]) == approx(1.8171205928321)


if __name__ == "__main__":
    main()
