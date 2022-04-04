# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for compiler_gym.spaces.Reward."""
from copy import deepcopy

import pytest

from compiler_gym.spaces import Reward
from tests.test_main import main


def test_reward_id_backwards_compatibility():
    """Test that Reward.id is backwards compatible with Reward.name.

    See: github.com/facebookresearch/CompilerGym/issues/381
    """
    with pytest.warns(DeprecationWarning, match="renamed `name`"):
        reward = Reward(id="foo")

    assert reward.id == "foo"
    assert reward.name == "foo"


def test_equal():
    reward = Reward(
        name="test_reward",
        observation_spaces=["a", "b"],
        default_value=5,
        min=-10,
        max=10,
        default_negates_returns=True,
        success_threshold=3,
        deterministic=False,
        platform_dependent=True,
    )
    assert reward == deepcopy(reward)
    assert reward == "test_reward"


def test_not_equal():
    reward = Reward(
        name="test_reward",
        observation_spaces=["a", "b"],
        default_value=5,
        min=-10,
        max=10,
        default_negates_returns=True,
        success_threshold=3,
        deterministic=False,
        platform_dependent=True,
    )
    reward2 = deepcopy(reward)
    reward2.name = "test_reward_2"
    assert reward != reward2
    assert reward != "test_reward_2"


if __name__ == "__main__":
    main()
