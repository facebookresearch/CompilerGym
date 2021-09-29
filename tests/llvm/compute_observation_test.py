# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the compute_observation() function."""
from pathlib import Path

import networkx.algorithms.isomorphism
import pytest

from compiler_gym.envs.llvm import LlvmEnv, compute_observation
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_invalid_observation_space_name(env: LlvmEnv, tmpdir):
    tmpdir = Path(tmpdir)
    env.reset()
    env.write_bitcode(tmpdir / "ir.bc")
    space = env.observation.spaces["Ir"]
    space.id = "NotARealName"

    with pytest.raises(
        ValueError, match="Invalid observation space name: NOT_A_REAL_NAME"
    ):
        compute_observation(space, tmpdir / "ir.bc")


def test_missing_file(env: LlvmEnv, tmpdir):
    tmpdir = Path(tmpdir)
    env.reset()

    with pytest.raises(FileNotFoundError, match=str(tmpdir / "ir.bc")):
        compute_observation(env.observation.spaces["Ir"], tmpdir / "ir.bc")


def test_timeout_expired(env: LlvmEnv, tmpdir):
    tmpdir = Path(tmpdir)
    env.reset(benchmark="cbench-v1/jpeg-c")  # larger benchmark
    env.write_bitcode(tmpdir / "ir.bc")
    space = env.observation.spaces["Programl"]

    with pytest.raises(
        TimeoutError, match="Failed to compute Programl observation in 0.1 seconds"
    ):
        compute_observation(space, tmpdir / "ir.bc", timeout=0.1)


@pytest.mark.parametrize(
    "observation_space", ["Ir", "IrInstructionCount", "ObjectTextSizeBytes"]
)
def test_observation_equivalence(env: LlvmEnv, tmpdir, observation_space: str):
    """Test that compute_observation() produces the same result as the environment."""
    tmpdir = Path(tmpdir)
    env.reset()
    env.write_bitcode(tmpdir / "ir.bc")

    observation = compute_observation(
        env.observation.spaces[observation_space], tmpdir / "ir.bc"
    )
    assert observation == env.observation[observation_space]


def test_observation_programl_equivalence(env: LlvmEnv, tmpdir):
    """Test that compute_observation() produces the same result as the environment."""
    tmpdir = Path(tmpdir)
    env.reset()
    env.write_bitcode(tmpdir / "ir.bc")

    G = compute_observation(env.observation.spaces["Programl"], tmpdir / "ir.bc")
    networkx.algorithms.isomorphism.is_isomorphic(G, env.observation.Programl())


if __name__ == "__main__":
    main()
