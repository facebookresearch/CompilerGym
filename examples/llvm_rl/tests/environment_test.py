# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from llvm_rl.model.environment import Environment


def test_basic_environment_config():
    model = Environment(id="llvm-ic-v0", max_episode_steps=3)
    with model.make_env() as env:
        assert env.spec.id == "llvm-ic-v0"
        assert env.reward_space == "IrInstructionCountOz"

        # Test max episode steps:
        env.reset()
        _, _, done, _ = env.step(env.action_space.sample())  # step 1
        assert not done

        _, _, done, _ = env.step(env.action_space.sample())  # step 2
        assert not done

        _, _, done, _ = env.step(env.action_space.sample())  # step 3
        assert done


def test_reward_and_observation_space():
    model = Environment(
        id="llvm-ic-v0",
        max_episode_steps=3,
        observation_space="Ir",
        reward_space="ObjectTextSizeBytes",
    )
    with model.make_env() as env:
        assert env.reward_space == "ObjectTextSizeBytes"
        assert env.observation_space_spec.id == "Ir"


def test_wrappers():
    model = Environment(
        id="llvm-ic-v0",
        max_episode_steps=3,
        wrappers=[
            {
                "wrapper": "ConstrainedCommandline",
                "args": {"flags": ["-mem2reg", "-reg2mem"]},
            }
        ],
    )
    with model.make_env() as env:
        assert env.action_space.flags == ["-mem2reg", "-reg2mem"]
        assert env.action_space.n == 2
