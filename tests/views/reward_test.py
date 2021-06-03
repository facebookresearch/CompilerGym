# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/views."""
import pytest

from compiler_gym.views import RewardView
from tests.test_main import main


class MockReward:
    def __init__(self, id, ret=None):
        self.id = id
        self.ret = list(reversed(ret or []))
        self.observation_spaces = []

    def update(self, *args, **kwargs):
        ret = self.ret[-1]
        del self.ret[-1]
        return ret


class MockObservationView:
    pass


def test_empty_space():
    reward = RewardView([], MockObservationView())
    with pytest.raises(ValueError) as ctx:
        _ = reward["foo"]
    assert str(ctx.value) == "No reward spaces"


def test_invalid_reward_name():
    reward = RewardView([MockReward(id="foo")], MockObservationView())
    with pytest.raises(KeyError):
        _ = reward["invalid"]


def test_reward_values():
    spaces = [
        MockReward(id="codesize", ret=[-5]),
        MockReward(id="runtime", ret=[10]),
    ]
    reward = RewardView(spaces, MockObservationView())

    value = reward["codesize"]
    assert value == -5

    value = reward["runtime"]
    assert value == 10


def test_reward_values_bound_methods():
    spaces = [
        MockReward(id="codesize", ret=[-5]),
        MockReward(id="runtime", ret=[10]),
    ]
    reward = RewardView(spaces, MockObservationView())

    value = reward.codesize()
    assert value == -5

    value = reward.runtime()
    assert value == 10


if __name__ == "__main__":
    main()
