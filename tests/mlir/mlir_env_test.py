# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the MLIR CompilerGym environments."""
from numbers import Real

import gym
import numpy as np
import pytest
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env

import compiler_gym
from compiler_gym.envs import CompilerEnv, mlir
from compiler_gym.envs.mlir import MlirEnv
from compiler_gym.service.connection import CompilerGymServiceConnection
from compiler_gym.spaces import (
    Box,
    Dict,
    Discrete,
    NamedDiscrete,
    Permutation,
    Scalar,
    SpaceSequence,
)
from compiler_gym.spaces import Tuple as TupleSpace
from compiler_gym.wrappers.mlir import MlirRlWrapperEnv, convert_action
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.common", "tests.pytest_plugins.mlir"]


@pytest.fixture(scope="function", params=["local", "service"])
def env(request) -> CompilerEnv:
    """Create an MLIR environment."""
    if request.param == "local":
        with gym.make("mlir-v0") as env:
            yield env
    else:
        service = CompilerGymServiceConnection(mlir.MLIR_SERVICE_BINARY)
        try:
            with MlirEnv(service=service.connection.url) as env:
                yield env
        finally:
            service.close()


def test_service_version(env: MlirEnv):
    assert env.version == compiler_gym.__version__


def test_compiler_version(env: MlirEnv):
    assert env.compiler_version.startswith("LLVM 14.")


def test_action_spaces_names(env: MlirEnv):
    assert {a.name for a in env.action_spaces} == {"MatrixMultiplication"}


def test_action_space(env: MlirEnv):
    expected_action_space = SpaceSequence(
        name="MatrixMultiplication",
        size_range=[1, 4],
        space=Dict(
            name=None,
            spaces={
                "tile_options": Dict(
                    name=None,
                    spaces={
                        "interchange_vector": Permutation(
                            name=None,
                            scalar_range=Scalar(name=None, min=0, max=2, dtype=int),
                        ),
                        "tile_sizes": Box(
                            name=None,
                            low=np.array([1] * 3, dtype=int),
                            high=np.array([2 ** 32] * 3, dtype=int),
                            dtype=np.int64,
                        ),
                        "promote": Scalar(name=None, min=False, max=True, dtype=bool),
                        "promote_full_tile": Scalar(
                            name=None, min=False, max=True, dtype=bool
                        ),
                        "loop_type": NamedDiscrete(
                            name=None,
                            items=["loops", "affine_loops"],
                        ),
                    },
                ),
                "vectorize_options": Dict(
                    name=None,
                    spaces={
                        "vectorize_to": NamedDiscrete(
                            name=None,
                            items=["dot", "matmul", "outer_product"],
                        ),
                        "vector_transfer_split": NamedDiscrete(
                            name=None,
                            items=["none", "linalg_copy", "vector_transfer"],
                        ),
                        "unroll_vector_transfers": Scalar(
                            name=None,
                            min=False,
                            max=True,
                            dtype=bool,
                        ),
                    },
                ),
            },
        ),
    )
    assert expected_action_space == env.action_space


def test_set_observation_space_from_spec(env: MlirEnv):
    env.observation_space = env.observation.spaces["Runtime"]
    obs = env.observation_space

    env.observation_space = "Runtime"
    assert env.observation_space == obs


def test_set_reward_space_from_spec(env: MlirEnv):
    env.reward_space = env.reward.spaces["IsRunnable"]
    reward = env.reward_space

    env.reward_space = "IsRunnable"
    assert env.reward_space == reward


def test_mlir_rl_wrapper_env_action_space(env: MlirEnv):
    wrapper_env = MlirRlWrapperEnv(env)
    action_space = wrapper_env.action_space
    tile_size = NamedDiscrete(
        name=None,
        items=["1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024"],
    )
    expected_subspace = Dict(
        name=None,
        spaces={
            "tile_options": Dict(
                name=None,
                spaces={
                    "interchange_vector": Discrete(name=None, n=6),
                    "tile_sizes": TupleSpace(
                        name=None, spaces=[tile_size, tile_size, tile_size]
                    ),
                    "promote": NamedDiscrete(name=None, items=["False", "True"]),
                    "promote_full_tile": NamedDiscrete(
                        name=None, items=["False", "True"]
                    ),
                    "loop_type": NamedDiscrete(
                        name=None,
                        items=["loops", "affine_loops"],
                    ),
                },
            ),
            "vectorize_options": Dict(
                name=None,
                spaces={
                    "vectorize_to": NamedDiscrete(
                        name=None, items=["dot", "matmul", "outer_product"]
                    ),
                    "vector_transfer_split": NamedDiscrete(
                        name=None,
                        items=["none", "linalg_copy", "vector_transfer"],
                    ),
                    "unroll_vector_transfers": NamedDiscrete(
                        name=None, items=["False", "True"]
                    ),
                },
            ),
        },
    )
    assert action_space[0] == expected_subspace
    for i in range(1, 4):
        assert action_space[i]["is_present"] == NamedDiscrete(
            name=None, items=["False", "True"]
        )
        assert action_space[i]["space"] == expected_subspace


def test_convert_action():
    action = [
        {
            "tile_options": {
                "interchange_vector": 5,
                "tile_sizes": [1, 3, 9],
                "promote": 1,
                "promote_full_tile": 0,
                "loop_type": 1,
            },
            "vectorize_options": {
                "vectorize_to": 2,
                "vector_transfer_split": 1,
                "unroll_vector_transfers": 1,
            },
        },
        {"is_present": 0},
    ]
    converted_action = convert_action(action)

    expected_action = [
        {
            "tile_options": {
                "interchange_vector": np.array([2, 1, 0], dtype=int),
                "tile_sizes": [2, 8, 512],
                "promote": True,
                "promote_full_tile": False,
                "loop_type": 1,
            },
            "vectorize_options": {
                "vectorize_to": 2,
                "vector_transfer_split": 1,
                "unroll_vector_transfers": True,
            },
        }
    ]

    assert len(converted_action) == len(expected_action)
    assert len(converted_action[0]) == len(expected_action[0])
    assert len(converted_action[0]["tile_options"]) == len(
        expected_action[0]["tile_options"]
    )
    assert len(converted_action[0]["vectorize_options"]) == len(
        expected_action[0]["vectorize_options"]
    )


def test_mlir_rl_wrapper_env_observation_space(env: MlirEnv):
    wrapper_env = MlirRlWrapperEnv(env)
    observation_space = wrapper_env.observation_space
    assert observation_space == Box(
        name="Runtime", shape=[1], low=0, high=np.inf, dtype=float
    )


def test_mlir_rl_wrapper_env_step(env: MlirEnv):
    wrapper_env = MlirRlWrapperEnv(env)
    action_space = wrapper_env.action_space
    action_space.seed(123)
    action = action_space.sample()
    print(action)
    observation, reward, done, _ = wrapper_env.step(action)
    assert isinstance(observation, np.ndarray)
    assert np.array_equal(observation.shape, [1])
    assert observation[0] > 0
    assert isinstance(reward, Real)
    assert observation[0] == -reward
    assert isinstance(done, bool)
    assert done


def test_mlir_rl_wrapper_env_reset(env: MlirEnv):
    wrapper_env = MlirRlWrapperEnv(env)
    action_space = wrapper_env.action_space
    action_space.seed(123)
    observation = wrapper_env.reset()
    assert isinstance(observation, np.ndarray)
    assert np.array_equal(observation.shape, [1])
    assert observation[0] == 0


def test_ppo_train_smoke():
    register_env(
        "mlir_env", lambda env_config: MlirRlWrapperEnv(env=gym.make("mlir-v0"))
    )
    config = {
        "env": "mlir_env",
        "framework": "torch",
        # Tweak the default model provided automatically by RLlib,
        # given the environment's observation- and action spaces.
        "model": {
            "fcnet_hiddens": [2, 2],
            "fcnet_activation": "relu",
        },
        "num_workers": 0,  # local worker only
        "train_batch_size": 2,
        "sgd_minibatch_size": 1,
        "num_sgd_iter": 1,
        "rollout_fragment_length": 2,
    }
    ray.init(local_mode=True)  # Runs PPO training in the same process
    trainer = PPOTrainer(config=config)
    trainer.train()
    ray.shutdown()


if __name__ == "__main__":
    main()
