# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/spaces:named_discrete."""
import pytest

from compiler_gym.spaces import NamedDiscrete
from tests.test_main import main


def test_empty_space():
    space = NamedDiscrete([])
    assert space.n == 0
    assert space.names == []


def test_invalid_name_lookup():
    space = NamedDiscrete(["foo"])
    with pytest.raises(ValueError):
        _ = space["bar"]


def test_space_size():
    space = NamedDiscrete(["a", "b", "c"])
    assert space.n == 3


def test_name_lookup():
    space = NamedDiscrete(["a", "b", "c"])
    assert space["a"] == 0
    assert space["b"] == 1
    assert space["c"] == 2


def test_repr():
    space = NamedDiscrete(["foo", "bar"])
    assert str(space) == "NamedDiscrete([foo, bar])"


def test_to_string():
    space = NamedDiscrete(["foo", "bar"])
    assert space.to_string(0) == "foo"
    assert space.to_string([0]) == "foo"
    assert space.to_string([0, 0, 1]) == "foo foo bar"


if __name__ == "__main__":
    main()
