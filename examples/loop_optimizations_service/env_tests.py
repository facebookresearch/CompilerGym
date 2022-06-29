# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the unrolling CompilerGym service example."""
import subprocess
from pathlib import Path

import gym
import numpy as np
import pytest

import compiler_gym
import examples.loop_optimizations_service as loop_optimizations_service
from compiler_gym.envs import CompilerEnv
from compiler_gym.errors import SessionNotFound
from compiler_gym.service.client_service_compiler_env import ClientServiceCompilerEnv
from compiler_gym.spaces import ActionSpace, Dict, NamedDiscrete, Scalar, Sequence
from compiler_gym.third_party.autophase import AUTOPHASE_FEATURE_NAMES
from tests.test_main import main


@pytest.fixture(scope="function")
def env() -> CompilerEnv:
    """Text fixture that yields an environment."""
    with gym.make("loops-opt-py-v0") as env_:
        yield env_


@pytest.fixture(scope="module")
def bin() -> Path:
    return loop_optimizations_service.LOOPS_OPT_PY_SERVICE_BINARY


def test_invalid_arguments(bin: Path):
    """Test that running the binary with unrecognized arguments is an error."""

    def run(cmd):
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
        )
        stdout, stderr = p.communicate(timeout=10)
        return p.returncode, stdout, stderr

    returncode, _, stderr = run([str(bin), "foobar"])
    assert "ERROR:" in stderr
    assert "'foobar'" in stderr
    assert returncode == 1

    returncode, _, stderr = run([str(bin), "--foobar"])
    # C++ and python flag parsing library emit slightly different error
    # messages.
    assert "ERROR:" in stderr or "FATAL" in stderr
    assert "'foobar'" in stderr
    assert returncode == 1


def test_versions(env: ClientServiceCompilerEnv):
    """Tests the GetVersion() RPC endpoint."""
    assert env.version == compiler_gym.__version__
    assert env.compiler_version == "1.0.0"


def test_action_space(env: CompilerEnv):
    """Test that the environment reports the service's action spaces."""
    assert env.action_spaces == [
        ActionSpace(
            NamedDiscrete(
                name="loop-opt",
                items=[
                    "--loop-unroll --unroll-count=2",
                    "--loop-unroll --unroll-count=4",
                    "--loop-unroll --unroll-count=8",
                    "--loop-unroll --unroll-count=16",
                    "--loop-unroll --unroll-count=32",
                    "--loop-vectorize -force-vector-width=2",
                    "--loop-vectorize -force-vector-width=4",
                    "--loop-vectorize -force-vector-width=8",
                    "--loop-vectorize -force-vector-width=16",
                    "--loop-vectorize -force-vector-width=32",
                ],
            )
        )
    ]


def test_observation_spaces(env: CompilerEnv):
    """Test that the environment reports the service's observation spaces."""
    env.reset()
    assert env.observation.spaces.keys() == {
        "ir",
        "Inst2vec",
        "Autophase",
        "AutophaseDict",
        "Programl",
        "runtime",
        "size",
    }
    assert env.observation.spaces["ir"].space == Sequence(
        name="ir",
        size_range=(0, np.iinfo(int).max),
        dtype=str,
    )
    assert env.observation.spaces["Inst2vec"].space == Sequence(
        name="Inst2vec",
        size_range=(0, np.iinfo(int).max),
        scalar_range=Scalar(
            name=None,
            min=np.iinfo(np.int64).min,
            max=np.iinfo(np.int64).max,
            dtype=np.int64,
        ),
        dtype=int,
    )
    assert env.observation.spaces["Autophase"].space == Sequence(
        name="Autophase",
        size_range=(len(AUTOPHASE_FEATURE_NAMES), len(AUTOPHASE_FEATURE_NAMES)),
        scalar_range=Scalar(
            name=None,
            min=np.iinfo(np.int64).min,
            max=np.iinfo(np.int64).max,
            dtype=np.int64,
        ),
        dtype=int,
    )
    assert env.observation.spaces["AutophaseDict"].space == Dict(
        name="AutophaseDict",
        spaces={
            name: Scalar(name=None, min=0, max=np.iinfo(np.int64).max, dtype=np.int64)
            for name in AUTOPHASE_FEATURE_NAMES
        },
    )
    assert env.observation.spaces["Programl"].space == Sequence(
        name="Programl",
        size_range=(0, np.iinfo(int).max),
        dtype=str,
    )
    assert env.observation.spaces["runtime"].space == Scalar(
        name="runtime", min=0, max=np.inf, dtype=float
    )
    assert env.observation.spaces["size"].space == Scalar(
        name="size", min=0, max=np.inf, dtype=float
    )


def test_reward_spaces(env: CompilerEnv):
    """Test that the environment reports the service's reward spaces."""
    env.reset()
    assert env.reward.spaces.keys() == {"runtime", "size"}


def test_step_before_reset(env: CompilerEnv):
    """Taking a step() before reset() is illegal."""
    with pytest.raises(SessionNotFound, match=r"Must call reset\(\) before step\(\)"):
        env.step(0)


def test_observation_before_reset(env: CompilerEnv):
    """Taking an observation before reset() is illegal."""
    with pytest.raises(SessionNotFound, match=r"Must call reset\(\) before step\(\)"):
        _ = env.observation["ir"]


def test_reward_before_reset(env: CompilerEnv):
    """Taking a reward before reset() is illegal."""
    with pytest.raises(SessionNotFound, match=r"Must call reset\(\) before step\(\)"):
        _ = env.reward["runtime"]


def test_reset_invalid_benchmark(env: CompilerEnv):
    """Test requesting a specific benchmark."""
    with pytest.raises(LookupError) as ctx:
        env.reset(benchmark="loops-opt-v0/foobar")
    assert str(ctx.value) == "Unknown program name"


def test_invalid_observation_space(env: CompilerEnv):
    """Test error handling with invalid observation space."""
    with pytest.raises(LookupError):
        env.observation_space = 100


def test_invalid_reward_space(env: CompilerEnv):
    """Test error handling with invalid reward space."""
    with pytest.raises(LookupError):
        env.reward_space = 100


def test_double_reset(env: CompilerEnv):
    """Test that reset() can be called twice."""
    env.reset()
    assert env.in_episode
    env.reset()
    assert env.in_episode


def test_Step_out_of_range(env: CompilerEnv):
    """Test error handling with an invalid action."""
    env.reset()
    with pytest.raises(ValueError) as ctx:
        env.step(100)
    assert str(ctx.value) == "Out-of-range"


def test_default_ir_observation(env: CompilerEnv):
    """Test default IR observation space."""
    env.observation_space = "ir"
    observation = env.reset()
    assert len(observation) > 0

    observation, reward, done, info = env.step(0)
    assert not done, info
    assert len(observation) > 0
    assert reward is None


def test_default_inst2vec_observation(env: CompilerEnv):
    """Test default inst2vec observation space."""
    env.observation_space = "Inst2vec"
    observation = env.reset()
    assert isinstance(observation, np.ndarray)
    assert len(observation) >= 0
    assert observation.dtype == np.int64
    assert all(obs >= 0 for obs in observation.tolist())


def test_default_autophase_observation(env: CompilerEnv):
    """Test default autophase observation space."""
    env.observation_space = "Autophase"
    observation = env.reset()
    assert isinstance(observation, np.ndarray)
    assert observation.shape == (len(AUTOPHASE_FEATURE_NAMES),)
    assert observation.dtype == np.int64
    assert all(obs >= 0 for obs in observation.tolist())


def test_default_autophase_dict_observation(env: CompilerEnv):
    """Test default autophase dict observation space."""
    env.observation_space = "AutophaseDict"
    observation = env.reset()
    assert isinstance(observation, dict)
    assert sorted(observation.keys()) == sorted(AUTOPHASE_FEATURE_NAMES)
    assert len(observation.values()) == len(AUTOPHASE_FEATURE_NAMES)
    assert all(obs >= 0 for obs in observation.values())


def test_default_programl_observation(env: CompilerEnv):
    """Test default observation space."""
    env.observation_space = "Programl"
    observation = env.reset()
    assert len(observation) > 0

    observation, reward, done, info = env.step(0)
    assert not done, info
    assert len(observation) > 0
    assert reward is None


def test_default_reward(env: CompilerEnv):
    """Test default reward space."""
    env.reward_space = "runtime"
    env.reset()
    observation, reward, done, info = env.step(0)
    assert not done, info
    assert observation is None
    assert reward is not None


def test_observations(env: CompilerEnv):
    """Test observation spaces."""
    env.reset()
    assert len(env.observation["ir"]) > 0
    assert all(env.observation["Inst2vec"] >= 0)
    assert all(env.observation["Autophase"] >= 0)
    assert len(env.observation["Programl"]) > 0


def test_rewards(env: CompilerEnv):
    """Test reward spaces."""
    env.reset()
    assert env.reward["runtime"] is not None


def test_benchmarks(env: CompilerEnv):
    assert list(env.datasets.benchmark_uris()) == [
        "benchmark://loops-opt-v0/add",
        "benchmark://loops-opt-v0/offsets1",
        "benchmark://loops-opt-v0/conv2d",
    ]


def test_fork(env: CompilerEnv):
    env.reset()
    env.step(0)
    env.step(1)
    other_env = env.fork()
    try:
        assert env.benchmark == other_env.benchmark
        assert other_env.actions == [0, 1]
    finally:
        other_env.close()


if __name__ == "__main__":
    main()
