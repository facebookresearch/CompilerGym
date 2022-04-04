# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from compiler_gym.spaces import Box
from tests.test_main import main


def test_equal():
    assert Box(low=0, high=1, name="test_box", shape=[1, 2], dtype=int) == Box(
        low=0, high=1, name="test_box", shape=[1, 2], dtype=int
    )
    assert Box(low=0, high=1, name="test_box", shape=[1, 2], dtype=int) == Box(
        low=0, high=1, name="test_box", shape=[1, 2], dtype=float
    )


def test_not_equal():
    box = Box(low=0, high=1, name="test_box", shape=[1, 2], dtype=int)
    assert box != Box(low=0, high=1, name="test_box_2", shape=[1, 2], dtype=int)
    assert box != Box(low=-1, high=1, name="test_box", shape=[1, 2], dtype=int)
    assert box != Box(low=0, high=2, name="test_box", shape=[1, 2], dtype=int)
    assert box != Box(low=0, high=1, name="test_box", shape=[1, 3], dtype=int)
    assert box != "not_a_box"


if __name__ == "__main__":
    main()
