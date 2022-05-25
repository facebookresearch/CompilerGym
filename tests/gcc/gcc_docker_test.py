# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the GCC CompilerGym service."""
import gym
import pytest

import compiler_gym.envs.gcc  # noqa register environments
from compiler_gym.errors import ServiceError
from tests.pytest_plugins.common import skip_on_ci, with_docker
from tests.test_main import main


@with_docker
def test_invalid_docker_image():
    with pytest.raises(ServiceError):
        gym.make("gcc-v0", gcc_bin="docker:not-a-valid-image")


@with_docker
def test_version_11():
    with gym.make("gcc-v0", gcc_bin="docker:gcc:11.2.0") as env:
        assert env.compiler_version == "gcc (GCC) 11.2.0"


@skip_on_ci
@with_docker
def test_version_10():
    with gym.make("gcc-v0", gcc_bin="docker:gcc:10.3.0") as env:
        assert env.compiler_version == "gcc (GCC) 10.3.0"
    with gym.make("gcc-v0", gcc_bin="docker:gcc:10.3") as env:
        assert env.compiler_version == "gcc (GCC) 10.3.0"
    with gym.make("gcc-v0", gcc_bin="docker:gcc:10") as env:
        assert env.compiler_version == "gcc (GCC) 10.3.0"


@skip_on_ci
@with_docker
def test_version_9():
    with gym.make("gcc-v0", gcc_bin="docker:gcc:9.4.0") as env:
        assert env.compiler_version == "gcc (GCC) 9.4.0"
    with gym.make("gcc-v0", gcc_bin="docker:gcc:9.4") as env:
        assert env.compiler_version == "gcc (GCC) 9.4.0"
    with gym.make("gcc-v0", gcc_bin="docker:gcc:9") as env:
        assert env.compiler_version == "gcc (GCC) 9.4.0"


@skip_on_ci
@with_docker
def test_version_8():
    with gym.make("gcc-v0", gcc_bin="docker:gcc:8.5.0") as env:
        assert env.compiler_version == "gcc (GCC) 8.5.0"
    with gym.make("gcc-v0", gcc_bin="docker:gcc:8.5") as env:
        assert env.compiler_version == "gcc (GCC) 8.5.0"
    with gym.make("gcc-v0", gcc_bin="docker:gcc:8") as env:
        assert env.compiler_version == "gcc (GCC) 8.5.0"


if __name__ == "__main__":
    main()
