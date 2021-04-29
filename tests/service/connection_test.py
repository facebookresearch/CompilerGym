# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/service:connection."""
import gym
import pytest

import compiler_gym.envs  # noqa Register LLVM environments.
from compiler_gym.service import (
    CompilerGymServiceConnection,
    ConnectionOpts,
    ServiceError,
)
from compiler_gym.service.proto import GetSpacesRequest
from tests.test_main import main


@pytest.fixture(scope="function")
def connection() -> CompilerGymServiceConnection:
    """Yields a connection to a local service."""
    env = gym.make("llvm-v0")
    try:
        yield env.service
    finally:
        env.close()


@pytest.fixture(scope="function")
def dead_connection() -> CompilerGymServiceConnection:
    """Yields a connection to a dead local service service."""
    env = gym.make("llvm-v0")
    try:
        # Kill the service.
        env.service.connection.process.kill()
        env.service.connection.process.communicate()

        yield env.service
    finally:
        env.close()


def test_create_invalid_options():
    with pytest.raises(TypeError) as ctx:
        CompilerGymServiceConnection("")
    assert str(ctx.value) == "No endpoint provided for service connection"


def test_create_channel_failed_subprocess(
    dead_connection: CompilerGymServiceConnection,
):
    with pytest.raises(
        (ServiceError, TimeoutError), match="Failed to create connection to localhost:"
    ):
        CompilerGymServiceConnection(
            f"{dead_connection.connection.url}",
            ConnectionOpts(
                init_max_seconds=1,
                init_max_attempts=2,
                rpc_init_max_seconds=0.1,
            ),
        )


def test_create_channel_failed_subprocess_rpc_timeout(
    dead_connection: CompilerGymServiceConnection,
):
    """Same as the above test, but RPC timeout is long enough that only a single
    attempt can be made.
    """
    with pytest.raises(OSError) as ctx:
        CompilerGymServiceConnection(
            f"{dead_connection.connection.url}",
            ConnectionOpts(
                init_max_seconds=0.1,
                init_max_attempts=2,
                rpc_init_max_seconds=1,
            ),
        )

    assert str(ctx.value).startswith("Failed to create connection to localhost:")
    assert " (1 attempt made)" in str(ctx.value)


def test_call_stub_invalid_type(connection: CompilerGymServiceConnection):
    with pytest.raises(TypeError) as ctx:
        connection(connection.stub.GetSpaces, int)
    assert str(ctx.value) == "Exception serializing request! Request type: type"


def test_call_stub_negative_timeout(connection: CompilerGymServiceConnection):
    with pytest.raises(TimeoutError) as ctx:
        connection(connection.stub.GetSpaces, GetSpacesRequest(), timeout=-10)
    assert str(ctx.value) == "Deadline Exceeded (-10.0 seconds)"


if __name__ == "__main__":
    main()
