# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import warnings

import gym
import numpy as np
import pytest
import torch
from flaky import flaky
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env

from compiler_gym.wrappers.mlir import make_mlir_rl_wrapper_env
from tests.test_main import main

# Ignore import deprecation warnings from ray.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import ray


@flaky(max_runs=3, min_passes=1)
@pytest.mark.filterwarnings(
    "ignore:`np\\.bool` is a deprecated alias for the builtin `bool`\\.",
    "ignore:Mean of empty slice",
    "ignore::ResourceWarning",
    "ignore:using `dtype=` in comparisons is only useful for `dtype=object`",
)
def test_rllib_ppo_smoke():
    ray.shutdown()
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    ray.init(local_mode=True)  # Runs PPO training in the same process
    register_env(
        "mlir_rl_env-v0",
        lambda env_config: make_mlir_rl_wrapper_env(env=gym.make("mlir-v0")),
    )
    config = {
        "env": "mlir_rl_env-v0",
        "framework": "torch",
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
    trainer = PPOTrainer(config=config)
    trainer.train()
    ray.shutdown()


if __name__ == "__main__":
    main()
