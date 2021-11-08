# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import pytest

import compiler_gym

from . import autophase_wrappers as wrappers


@pytest.fixture(scope="function")
def env():
    with compiler_gym.make("llvm-v0") as env:
        yield env


def test_AutophaseNormalizedFeatures(env):
    env = wrappers.AutophaseNormalizedFeatures(env)
    assert env.observation_space_spec.id == "Autophase"
    assert env.observation_space.shape == (56,)
    assert env.observation_space.dtype == np.float32


def test_ConcatActionsHistogram(env):
    env.observation_space = "Autophase"
    num_features = env.observation_space.shape[0]
    num_actions = env.action_space.n

    env = wrappers.ConcatActionsHistogram(env)
    env.reset()
    action = env.action_space.sample()
    obs, _, _, _ = env.step(action)
    assert env.observation_space.shape == (num_features + num_actions,)
    assert obs.shape == (num_features + num_actions,)


def test_AutophaseActionSpace(env):
    env = wrappers.AutophaseActionSpace(env)

    env.reset()
    env.step(env.action_space.sample())
    assert env.action_space.n == 42
