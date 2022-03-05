# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/spaces:sequence."""
import pytest

from compiler_gym.spaces import Scalar, Sequence, SpaceSequence
from tests.test_main import main


def test_sample():
    space = Sequence(name="test", size_range=(0, None), dtype=int)
    with pytest.raises(NotImplementedError):
        space.sample()


def test_str_contains():
    space = Sequence(name="test", size_range=(0, None), dtype=str)
    assert space.contains("Hello, world!")
    assert space.contains("")
    assert not space.contains([1, 2, 3])


def test_str_contains_too_long():
    space = Sequence(name="test", size_range=(0, 4), dtype=str)
    assert not space.contains("Hello, world!")
    assert space.contains("")
    assert not space.contains([1, 2, 3])


def test_str_contains_too_short():
    space = Sequence(name="test", size_range=(3, None), dtype=str)
    assert space.contains("Hello, world!")
    assert not space.contains("")
    assert not space.contains([1, 2, 3])


def test_int_contains():
    space = Sequence(name="test", size_range=(5, 5), dtype=int)
    assert not space.contains(list(range(4)))
    assert space.contains(list(range(5)))
    assert not space.contains(list(range(6)))


def test_contains_with_float_scalar_range():
    space = Sequence(
        name="test",
        size_range=(3, 3),
        dtype=float,
        scalar_range=Scalar(name="test", min=0, max=1, dtype=float),
    )
    assert space.contains([0.0, 0.0, 0.0])
    assert space.contains([0.1, 1.0, 0.5])
    assert not space.contains([0.0, 0.0, -1.0])  # out of bounds
    assert not space.contains([0.0, 0, 0.1])  # wrong dtype
    assert not space.contains([0.0, 0])  # wrong shape


def test_bytes_contains():
    space = Sequence(name="test", size_range=(0, None), dtype=bytes)
    assert space.contains(b"Hello, world!")
    assert space.contains(b"")
    assert not space.contains("Hello, world!")


def test_space_sequence_contains():
    subspace = Scalar(name="subspace", min=0, max=1, dtype=float)
    space_seq = SpaceSequence(name="seq", space=subspace, size_range=(0, 2))
    assert space_seq.contains([0.5, 0.6])
    assert not space_seq.contains(["not-a-number"])
    assert not space_seq.contains([2.0])
    assert not space_seq.contains([0.1, 0.2, 0.3])


if __name__ == "__main__":
    main()
