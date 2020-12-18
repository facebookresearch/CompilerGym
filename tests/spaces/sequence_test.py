# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/spaces:sequence."""
import pytest

from compiler_gym.spaces import Sequence
from tests.test_main import main


def test_sample():
    space = Sequence(size_range=(0, None), dtype=int)
    with pytest.raises(NotImplementedError):
        space.sample()


def test_str_contains():
    space = Sequence(size_range=(0, None), dtype=str)
    assert space.contains("Hello, world!")
    assert space.contains("")
    assert not space.contains([1, 2, 3])


def test_str_contains_too_long():
    space = Sequence(size_range=(0, 4), dtype=str)
    assert not space.contains("Hello, world!")
    assert space.contains("")
    assert not space.contains([1, 2, 3])


def test_str_contains_too_short():
    space = Sequence(size_range=(3, None), dtype=str)
    assert space.contains("Hello, world!")
    assert not space.contains("")
    assert not space.contains([1, 2, 3])


def test_int_contains():
    space = Sequence(size_range=(5, 5), dtype=int)
    assert not space.contains(list(range(4)))
    assert space.contains(list(range(5)))
    assert not space.contains(list(range(6)))


if __name__ == "__main__":
    main()
