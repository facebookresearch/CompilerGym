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
    with gym.make("llvm-v0") as env:
        yield env.service


@pytest.fixture(scope="function")
def dead_connection() -> CompilerGymServiceConnection:
    """Yields a connection to a dead local service service."""
    with gym.make("llvm-v0") as env:
        # Kill the service.
        env.service.connection.process.terminate()
        env.service.connection.process.communicate()

        yield env.service


def test_create_invalid_options():
    with pytest.raises(TypeError, match="No endpoint provided for service connection"):
        CompilerGymServiceConnection("")


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
    with pytest.raises(
        OSError,
        match=(
            r"Failed to create connection to localhost:\d+ after "
            r"[\d\.]+ seconds \(1 attempt made\)"
        ),
    ):
        CompilerGymServiceConnection(
            f"{dead_connection.connection.url}",
            ConnectionOpts(
                init_max_seconds=0.1,
                init_max_attempts=2,
                rpc_init_max_seconds=1,
            ),
        )


def test_call_stub_invalid_type(connection: CompilerGymServiceConnection):
    with pytest.raises(
        TypeError, match="Exception serializing request! Request type: type"
    ):
        connection(connection.stub.GetSpaces, int)


def test_call_stub_negative_timeout(connection: CompilerGymServiceConnection):
    with pytest.raises(TimeoutError, match=r"Deadline Exceeded \(-10.0 seconds\)"):
        connection(connection.stub.GetSpaces, GetSpacesRequest(), timeout=-10)


def test_ManagedConnection_repr(connection: CompilerGymServiceConnection):
    cnx = connection.connection
    assert (
        repr(cnx)
        == f"Connection to service at {cnx.url} running on PID {cnx.process.pid}"
    )

    # Kill the service.
    cnx.process.terminate()
    cnx.process.communicate()

    assert repr(cnx) == f"Connection to dead service at {cnx.url}"


if __name__ == "__main__":
    main()
