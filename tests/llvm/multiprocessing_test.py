# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the LLVM CompilerGym environments."""
import multiprocessing as mp
import sys
from typing import List

import gym
import pytest
from flaky import flaky

from compiler_gym.envs import LlvmEnv
from tests.pytest_plugins.common import macos_only
from tests.test_main import main


def process_worker(env_name: str, benchmark: str, actions: List[int], queue: mp.Queue):
    assert actions
    with gym.make(env_name) as env:
        env.reset(benchmark=benchmark)

        for action in actions:
            observation, reward, done, info = env.step(action)
            assert not done

        queue.put((observation, reward, done, info))


def process_worker_with_env(env: LlvmEnv, actions: List[int], queue: mp.Queue):
    assert actions

    for action in actions:
        observation, reward, done, info = env.step(action)
        assert not done

    queue.put((env, observation, reward, done, info))


@flaky  # Test contains timeouts.
def test_running_environment_in_background_process():
    """Test launching and running an LLVM environment in a background process."""
    queue = mp.Queue(maxsize=3)
    process = mp.Process(
        target=process_worker,
        args=("llvm-autophase-ic-v0", "cbench-v1/crc32", [0, 0, 0], queue),
    )
    process.start()
    try:
        process.join(timeout=60)
        result = queue.get(timeout=60)
        observation, reward, done, info = result

        assert not done
        assert observation is not None
        assert isinstance(reward, float)
        assert info
    finally:
        # kill() was added in Python 3.7.
        if sys.version_info >= (3, 7, 0):
            process.kill()
        else:
            process.terminate()
        process.join(timeout=60)


@macos_only
@pytest.mark.skipif(sys.version_info < (3, 8, 0), reason="Py >= 3.8 only")
def test_moving_environment_to_background_process_macos():
    """Test moving an LLVM environment to a background process."""
    queue = mp.Queue(maxsize=3)

    with gym.make("llvm-autophase-ic-v0") as env:
        env.reset(benchmark="cbench-v1/crc32")

        process = mp.Process(
            target=process_worker_with_env, args=(env, [0, 0, 0], queue)
        )

        # Moving an environment to a background process is not supported because
        # we are using a subprocess.Popen() to manage the service binary, which
        # doesn't support pickling.
        with pytest.raises(TypeError):
            process.start()


def test_port_collision_test():
    """Test that attempting to connect to a port that is already in use succeeds."""
    with gym.make("llvm-autophase-ic-v0") as env_a:
        env_a.reset(benchmark="cbench-v1/crc32")

        with LlvmEnv(service=env_a.service.connection.url) as env_b:
            env_b.reset(benchmark="cbench-v1/crc32")


if __name__ == "__main__":
    main()
