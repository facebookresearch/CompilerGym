# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/views."""
import pytest

from compiler_gym.service.proto import (
    Reward,
    RewardRequest,
    RewardSpace,
    ScalarLimit,
    ScalarRange,
)
from compiler_gym.views import RewardView
from tests.test_main import main


class MockGetReward(object):
    """Mock for the get_reward callack of RewardView."""

    def __init__(self, ret=None):
        self.called_reward_spaces = []
        self.ret = list(reversed(ret or []))

    def __call__(self, request: RewardRequest):
        self.called_reward_spaces.append(request.reward_space)
        ret = self.ret[-1]
        del self.ret[-1]
        return ret


def test_empty_space():
    with pytest.raises(ValueError) as ctx:
        RewardView(MockGetReward(), [])
    assert str(ctx.value) == "No reward spaces"


def test_invalid_reward_name():
    spaces = [RewardSpace(name="codesize", range=ScalarRange(min=ScalarLimit(value=0)))]
    reward = RewardView(MockGetReward(), spaces)
    with pytest.raises(KeyError):
        _ = reward["invalid"]


def test_reward_values():
    spaces = [
        RewardSpace(name="codesize", range=ScalarRange(max=ScalarLimit(value=0))),
        RewardSpace(name="runtime", range=ScalarRange(min=ScalarLimit(value=0))),
    ]
    mock = MockGetReward(ret=[Reward(reward=-5), Reward(reward=10)])
    reward = RewardView(mock, spaces)

    value = reward["codesize"]
    assert value == -5

    value = reward["runtime"]
    assert value == 10

    # Check that the correct reward_space_list indices were used.
    assert mock.called_reward_spaces == [0, 1]


if __name__ == "__main__":
    main()
