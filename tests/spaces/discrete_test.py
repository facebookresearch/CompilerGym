# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from compiler_gym.spaces import Discrete
from tests.test_main import main


def test_equal():
    assert Discrete(2, name="test_discrete") == Discrete(2, name="test_discrete")


def test_not_equal():
    discrete = Discrete(2, name="test_discrete")
    assert discrete != Discrete(3, name="test_discrete")
    assert discrete != Discrete(2, name="test_discrete_2")
    assert discrete != "not_a_discrete"


if __name__ == "__main__":
    main()
