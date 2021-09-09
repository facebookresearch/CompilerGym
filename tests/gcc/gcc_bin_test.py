# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the GCC CompilerGym service."""
import gym
import pytest

import compiler_gym.envs.gcc  # noqa register environments
from compiler_gym.service import ServiceError
from tests.pytest_plugins.common import with_docker
from tests.pytest_plugins.gcc import with_system_gcc, without_system_gcc
from tests.test_main import main


def test_missing_gcc_bin():
    with pytest.raises(ServiceError):
        gym.make("gcc-v0", gcc_bin="not-a-real-file")


def test_invalid_gcc_bin():
    with pytest.raises(ServiceError):
        gym.make("gcc-v0", gcc_bin="false")


@with_docker
def test_invalid_docker_image():
    with pytest.raises(ServiceError):
        gym.make("gcc-v0", gcc_bin="docker:not-a-valid-image")


@with_system_gcc
def test_system_gcc():
    env = gym.make("gcc-v0", gcc_bin="gcc")
    assert "gcc" in env.compiler_version


@without_system_gcc
def test_missing_system_gcc():
    with pytest.raises(ServiceError):
        gym.make("gcc-v0", gcc_bin="gcc")


if __name__ == "__main__":
    main()
