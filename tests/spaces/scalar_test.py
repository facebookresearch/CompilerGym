# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/spaces:scalar."""
from compiler_gym.spaces import Scalar
from tests.test_main import main


def test_sample():
    space = Scalar(min=-10, max=10, dtype=int)
    x = space.sample()
    assert isinstance(x, int)
    assert -10 <= x <= 10


def test_int_contains():
    space = Scalar(min=-10, max=10, dtype=int)
    assert space.contains(-10)
    assert not space.contains(-11)
    assert not space.contains(0.5)


def test_int_contains_no_upper_bound():
    space = Scalar(min=0, max=None, dtype=int)
    assert space.contains(0)
    assert not space.contains(-1)
    assert space.contains(1000)


def test_equality():
    space_a = Scalar(min=0, max=None, dtype=int)
    space_b = Scalar(min=0, max=None, dtype=int)
    assert space_a == space_b


def test_dtype_inequality():
    space_a = Scalar(min=0, max=None, dtype=int)
    space_b = Scalar(min=0, max=None, dtype=float)
    assert space_a != space_b


def test_upper_bound_inequality():
    space_a = Scalar(min=0, max=None, dtype=int)
    space_b = Scalar(min=0, max=5, dtype=int)
    assert space_a != space_b


def test_lower_bound_inequality():
    space_a = Scalar(min=0, max=None, dtype=int)
    space_b = Scalar(min=None, max=None, dtype=int)
    assert space_a != space_b


if __name__ == "__main__":
    main()
