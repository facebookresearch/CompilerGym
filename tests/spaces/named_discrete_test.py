# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/spaces:named_discrete."""
import pytest

from compiler_gym.spaces import NamedDiscrete
from tests.test_main import main


def test_empty_space():
    with pytest.raises(ValueError, match="No values for discrete space"):
        NamedDiscrete([], name="test")


def test_invalid_name_lookup():
    space = NamedDiscrete(["foo"], name="test")
    with pytest.raises(ValueError):
        _ = space["bar"]


def test_space_size():
    space = NamedDiscrete(["a", "b", "c"], name="test")
    assert space.n == 3


def test_name_lookup():
    space = NamedDiscrete(["a", "b", "c"], name="test")
    assert space["a"] == 0
    assert space["b"] == 1
    assert space["c"] == 2


def test_repr():
    space = NamedDiscrete(["foo", "bar"], name="test")
    assert str(space) == "NamedDiscrete([foo, bar])"


def test_to_string():
    space = NamedDiscrete(["foo", "bar"], name="test")
    assert space.to_string(0) == "foo"
    assert space.to_string([0]) == "foo"
    assert space.to_string([0, 0, 1]) == "foo foo bar"


def test_equal():
    assert NamedDiscrete(["a", "b"], name="test_named_discrete") == NamedDiscrete(
        ["a", "b"], name="test_named_discrete"
    )


def test_not_equal():
    named_discrete = NamedDiscrete(["a", "b"], name="test_named_discrete")
    assert named_discrete != NamedDiscrete(["a", "bb"], name="test_named_discrete")
    assert named_discrete != NamedDiscrete(["a", "b", "c"], name="test_named_discrete")
    assert named_discrete != NamedDiscrete(["a", "b"], name="test_named_discrete_2")
    assert named_discrete != "not_a_named_discrete"


if __name__ == "__main__":
    main()
