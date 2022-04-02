# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from gym.spaces import Discrete

from compiler_gym.spaces import Dict


def test_equal():
    assert Dict({"a": Discrete(2), "b": Discrete(3)}, name="test_dict") == Dict(
        {"a": Discrete(2), "b": Discrete(3)}, name="test_dict"
    )


def test_not_equal():
    dict_space = Dict({"a": Discrete(2), "b": Discrete(3)}, name="test_dict")
    assert dict_space != Dict({"a": Discrete(2), "c": Discrete(3)}, name="test_dict")
    assert dict_space != Dict({"a": Discrete(2)}, name="test_dict")
    assert dict_space != Dict({"a": Discrete(2), "b": Discrete(3)}, name="test_dict_2")
    assert dict_space != "not_a_dict"
