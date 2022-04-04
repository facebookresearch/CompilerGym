# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from gym.spaces import Discrete

from compiler_gym.spaces import Tuple
from tests.test_main import main


def test_equal():
    assert Tuple([Discrete(2), Discrete(3)], name="test_tuple") == Tuple(
        [Discrete(2), Discrete(3)], name="test_tuple"
    )


def test_not_equal():
    tuple_space = Tuple([Discrete(2), Discrete(3)], name="test_tuple")
    assert tuple_space != Tuple([Discrete(3), Discrete(3)], name="test_tuple")
    assert tuple_space != Tuple([Discrete(2)], name="test_tuple")
    assert tuple_space != Tuple([Discrete(2), Discrete(3)], name="test_tuple_2")
    assert tuple_space != "not_a_tuple"


if __name__ == "__main__":
    main()
