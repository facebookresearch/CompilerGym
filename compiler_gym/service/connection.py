# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module contains the logic for connecting to services."""
import logging
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import sleep, time
from typing import List, NamedTuple, Optional, TypeVar, Union

import grpc

from compiler_gym.service.proto import (
    ActionSpace,
    CompilerGymServiceStub,
    GetSpacesReply,
    GetSpacesRequest,
    ObservationSpace,
    RewardSpace,
)
from compiler_gym.util.runfiles_path import cache_path, runfiles_path

GRPC_CHANNEL_OPTIONS = [
    # Raise the default inbound message filter from 4MB.
    ("grpc.max_receive_message_length", 100 * 1024 * 1024),
    # Spurious error UNAVAILABLE "Trying to connect an http1.x server".
    # https://putridparrot.com/blog/the-unavailable-trying-to-connect-an-http1-x-server-grpc-error/
    ("grpc.enable_http_proxy", 0),
]


class ConnectionOpts(NamedTuple):
    """The options used to configure a connection to a service."""

    rpc_call_max_seconds: float = 300
    """The maximum number of seconds to wait for an RPC method call to succeed."""

    init_max_seconds: float = 10
    """The maximum number of seconds to spend attempting to establish a
    connection to the service before failing.
    """

    init_max_attempts: int = 5
    """The maximum number of attempts to make to establish a connection to the
    service before failing.
    """

    local_service_port_init_max_seconds: float = 10
    """The maximum number of seconds to wait for a local service to write the port.txt file."""

    local_service_exit_max_seconds: float = 10
    """The maximum number of seconds to wait for a local service to terminate on close."""

    rpc_init_max_seconds: float = 3
    """The maximum number of seconds to wait for an RPC connection to establish."""


class ServiceError(Exception):
    """Error raised from the service."""


class ServiceInitError(ServiceError, OSError):
    """Error raised if the service fails to initialize."""


class ServiceTransportError(ServiceError, OSError):
    """Error that is raised if communication with the service fails."""


class ServiceIsClosed(ServiceError, TypeError):
    """Error that is raised if trying to interact with a closed service."""


Request = TypeVar("Request")
Reply = TypeVar("Reply")

if sys.version_info > (3, 8, 0):
    from typing import Protocol

    class StubMethod(Protocol):
        """Type annotation for an RPC stub method that accepts a request message
        and returns a reply.
        """

        Request = TypeVar("Request")
        Reply = TypeVar("Reply")

        def __call__(self, a: Request, timeout: float) -> Reply:
            ...


else:
    # Legacy support for Python < 3.8.
    from typing import Callable

    StubMethod = Callable[[Request], Reply]


class Connection(object):
    """Base class for service connections."""

    def __init__(self, channel, url: str):
        """Constructor. Don't instantiate this directly, use the subclasses.

        :param channel: The RPC channel to use.
        :param url: The URL of the RPC service.
        """
        self.channel = channel
        self.url = url
        self.stub = CompilerGymServiceStub(self.channel)
        self.spaces: GetSpacesReply = self(self.stub.GetSpaces, GetSpacesRequest())

    def close(self):
        self.channel.close()

    def __call__(
        self,
        stub_method: StubMethod,
        request: Request,
        timeout: float = 60,
    ) -> Reply:
        """Call the service with the given arguments."""
        # House keeping note: if you modify the exceptions that this method
        # raises, please update the CompilerGymServiceConnection.__call__()
        # docstring.
        try:
            return stub_method(request, timeout=timeout)
        except ValueError as e:
            if str(e) == "Cannot invoke RPC on closed channel!":
                raise ServiceIsClosed(f"RPC communication failed with message: {e}")
            raise e
        except grpc.RpcError as e:
            # We raise "from None" to discard the gRPC stack trace, with the
            # remaining stack trace correctly pointing to the CompilerGym
            # calling code.
            if e.code() == grpc.StatusCode.INVALID_ARGUMENT:
                raise ValueError(e.details()) from None
            elif e.code() == grpc.StatusCode.UNIMPLEMENTED:
                raise NotImplementedError(e.details()) from None
            elif e.code() == grpc.StatusCode.NOT_FOUND:
                raise FileNotFoundError(e.details()) from None
            elif e.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                raise OSError(e.details()) from None
            elif e.code() == grpc.StatusCode.FAILED_PRECONDITION:
                raise TypeError(str(e.details())) from None
            elif e.code() == grpc.StatusCode.UNAVAILABLE:
                raise ServiceTransportError(f"{self.url} {e.details()}") from None
            elif (
                e.code() == grpc.StatusCode.INTERNAL
                and e.details() == "Exception serializing request!"
            ):
                raise TypeError(
                    f"{e.details()} Request type: {type(request).__name__}"
                ) from None
            elif e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                raise TimeoutError(f"{e.details()} ({timeout:.1f} seconds)") from None
            else:
                raise ServiceError(
                    f"RPC call returned status code {e.code()} and error `{e.details()}`"
                ) from None


def make_working_dir():
    """Make a working directory for a service. The calling code is responsible for
    removing this directory when done.
    """
    service_directory = cache_path("service")
    timestamp = datetime.now().isoformat()
    random_hash = random.getrandbits(32)
    working_dir = Path(service_directory / f"{timestamp}-{random_hash:08x}")
    (working_dir / "logs").mkdir(parents=True, exist_ok=False)
    return working_dir


class ManagedConnection(Connection):
    """A connection to a service using a managed subprocess."""

    def __init__(
        self,
        local_service_binary: Path,
        port_init_max_seconds: float,
        rpc_init_max_seconds: float,
        process_exit_max_seconds: float,
    ):
        """Constructor.

        :param local_service_binary: The path of the service binary.
        :raises TimeoutError: If fails to establish connection within a specified time limit.
        """
        self.process_exit_max_seconds = process_exit_max_seconds

        if not Path(local_service_binary).is_file():
            raise FileNotFoundError(f"File not found: {local_service_binary}")
        self.working_dir = make_working_dir()

        # Set environment variable COMPILER_GYM_SERVICE_ARGS to pass
        # additional arguments to the service.
        args = os.environ.get("COMPILER_GYM_SERVICE_ARGS", "")
        cmd = [
            str(local_service_binary),
            f"--port=0",
            f"--log_dir={self.working_dir}/logs",
            f"--working_dir={self.working_dir}",
            args,
        ]

        # Set the root of the runfiles directory.
        env = os.environ.copy()
        env["COMPILER_GYM_RUNFILES"] = str(runfiles_path("CompilerGym"))

        # Set environment variable COMPILER_GYM_SERVICE_DEBUG=1 to pipe
        # local service output to stderr. Set COMPILER_GYM_SERVICE_LOG_LEVEL=val
        # to set log level to <val>, where large values are increasingly
        # verbose.
        if os.environ.get("COMPILER_GYM_SERVICE_DEBUG", "0") != "0":
            cmd.append("--alsologtostderr")
        log_level = os.environ.get("COMPILER_GYM_SERVICE_LOG_LEVEL", "0")
        if log_level:
            cmd.append(f"-v={log_level}")

        logging.debug(f"$ {' '.join(cmd)}")
        self.process = subprocess.Popen(cmd, env=env)

        # Read the port from a file generated by the service.
        wait_secs = 0.1
        port_path = self.working_dir / "port.txt"
        end_time = time() + port_init_max_seconds
        while time() < end_time:
            returncode = self.process.poll()
            if returncode is not None:
                shutil.rmtree(self.working_dir)
                raise ServiceError(f"Service terminated with returncode: {returncode}")
            if port_path.is_file():
                with open(port_path) as f:
                    self.port = int(f.read().rstrip())
                break
            sleep(wait_secs)
            wait_secs *= 1.2
        else:
            self.process.kill()
            self.process.communicate(timeout=rpc_init_max_seconds)
            shutil.rmtree(self.working_dir)
            raise TimeoutError(
                "Service failed to produce port file after "
                f"{port_init_max_seconds:.1f} seconds"
            )

        url = f"localhost:{self.port}"

        wait_secs = 0.1
        attempts = 0
        end_time = time() + rpc_init_max_seconds
        while time() < end_time:
            try:
                channel = grpc.insecure_channel(
                    url,
                    options=GRPC_CHANNEL_OPTIONS,
                )
                channel_ready = grpc.channel_ready_future(channel)
                attempts += 1
                channel_ready.result(timeout=wait_secs)
                break
            except (grpc.FutureTimeoutError, grpc.RpcError) as e:
                logging.debug(f"Connection attempt {attempts} = {e}")
                wait_secs *= 1.2
        else:
            self.process.kill()
            self.process.communicate(timeout=process_exit_max_seconds)
            shutil.rmtree(self.working_dir)
            raise TimeoutError(
                "Failed to connect to RPC service after "
                f"{process_exit_max_seconds:.1f} seconds"
                f"({attempts} attempts made)"
            )

        super().__init__(channel, url)

    def close(self):
        """Terminate a local subprocess and close the connection."""
        self.process.kill()
        self.process.communicate(timeout=self.process_exit_max_seconds)
        shutil.rmtree(self.working_dir, ignore_errors=True)
        super().close()

    def __repr__(self):
        return f"{self.url} running on PID={self.process.pid}"


class UnmanagedConnection(Connection):
    """A connection to a service that is not managed by this process."""

    def __init__(self, url: str, rpc_init_max_seconds: float):
        """Constructor.

        :param url: The URL of the service to connect to.
        :raises TimeoutError: If fails to establish connection within a specified time limit.
        """

        wait_secs = 0.1
        attempts = 0
        end_time = time() + rpc_init_max_seconds
        while time() < end_time:
            try:
                channel = grpc.insecure_channel(
                    url,
                    options=GRPC_CHANNEL_OPTIONS,
                )
                channel_ready = grpc.channel_ready_future(channel)
                attempts += 1
                channel_ready.result(timeout=wait_secs)
                break
            except (grpc.FutureTimeoutError, grpc.RpcError) as e:
                logging.debug(f"Connection attempt {attempts} = {e}")
                wait_secs *= 1.2
        else:
            raise TimeoutError(
                f"Failed to connect to {url} after "
                f"{rpc_init_max_seconds:.1f} seconds "
                f"({attempts} attempts made)"
            )

        super().__init__(channel, url)

    def __repr__(self):
        return self.url


class CompilerGymServiceConnection(object):
    """A connection to a compiler gym service.

    There are two types of service connections: managed and unmanaged. The type
    of connection is determined by the endpoint. If a "host:port" URL is provided,
    an unmanaged connection is created. If the path of a file is provided, a
    managed connection is used. The difference between a managed and unmanaged
    connection is that with a managed connection, the lifecycle of the service
    if controlled by the client connection. That is, when a managed connection
    is created, a service subprocess is started by executing the specified path.
    When the connection is closed, the subprocess is terminated. With an
    unmanaged connection, if the service fails is goes offline, the client will
    fail.

    This class provides a common abstraction between the two types of connection,
    and provides a call method for invoking remote procedures on the service.

    Example usage of an unmanaged service connection:

    .. code-block:: python

        # Connect to a service running on localhost:8080. The user must
        # started a process running on port 8080.
        connection = CompilerGymServiceConnection("localhost:8080")
        # Invoke an RPC method.
        connection(connection.stub.StartEpisode, StartEpisodeRequest())
        # Close the connection. The service running on port 8080 is
        # left running.
        connection.close()

    Example usage of a managed service connection:

    .. code-block:: python

        # Start a subprocess using the binary located at /path/to/my/service.
        connection = CompilerGymServiceConnection(Path("/path/to/my/service"))
        # Invoke an RPC method.
        connection(connection.stub.StartEpisode, StartEpisodeRequest())
        # Close the connection. The subprocess is terminated.
        connection.close()

    :ivar stub: A CompilerGymServiceStub that can be used as the first argument
        to :py:meth:`__call__()` to specify an RPC
        method to call.
    :ivar action_spaces: A list of action spaces provided by the service.
    :ivar observation_spaces: A list of observation spaces provided by the
        service.
    :ivar reward_spaces: A list of reward spaces provided by the service.
    """

    def __init__(self, endpoint: Union[str, Path], opts: ConnectionOpts = None):
        """Constructor.

        :param endpoint: The connection endpoint. Either the URL of a service,
            e.g. "localhost:8080", or the path of a local service binary.
        :param opts: The connection options.
        :raises ValueError: If the provided options are invalid.
        :raises FileNotFoundError: In case opts.local_service_binary is not found.
        :raises TimeoutError: In case the service failed to start within
                opts.init_max_seconds seconds.
        """
        self.endpoint = endpoint
        self.opts = opts or ConnectionOpts()
        self.connection = None
        self.stub = None
        self._establish_connection()

        self.action_spaces: List[ActionSpace] = list(
            self.connection.spaces.action_space_list
        )
        self.observation_spaces: List[ObservationSpace] = list(
            self.connection.spaces.observation_space_list
        )
        self.reward_spaces: List[RewardSpace] = list(
            self.connection.spaces.reward_space_list
        )

    def _establish_connection(self) -> None:
        """Create and establish a connection."""
        self.connection = self._create_connection(self.endpoint, self.opts)
        self.stub = self.connection.stub

    @classmethod
    def _create_connection(
        cls, endpoint: Union[str, Path], opts: ConnectionOpts
    ) -> Connection:
        """Initialize the service connection, either by connecting to an RPC
        service or by starting a locally-managed subprocess.

        :param endpoint: The connection endpoint. Either the URL of a service,
            e.g. "localhost:8080", or the path of a local service binary.
        :param opts: The connection options.
        :raises ValueError: If the provided options are invalid.
        :raises FileNotFoundError: In case opts.local_service_binary is not found.
        :raises ServiceError: In case opts.init_max_attempts failures are made
            without successfully starting the connection.
        :raises TimeoutError: In case the service failed to start within
            opts.init_max_seconds seconds.
        """
        if not endpoint:
            raise TypeError("No endpoint provided for service connection")

        start_time = time()
        end_time = start_time + opts.init_max_seconds
        attempts = 0
        while time() < end_time and attempts < opts.init_max_attempts:
            attempts += 1
            try:
                if isinstance(endpoint, Path):
                    return ManagedConnection(
                        local_service_binary=endpoint,
                        process_exit_max_seconds=opts.local_service_exit_max_seconds,
                        rpc_init_max_seconds=opts.rpc_init_max_seconds,
                        port_init_max_seconds=opts.local_service_port_init_max_seconds,
                    )
                else:
                    return UnmanagedConnection(
                        url=endpoint, rpc_init_max_seconds=opts.rpc_init_max_seconds
                    )
            except (TimeoutError, ServiceError, NotImplementedError) as e:
                # Catch preventable errors so that we can retry:
                #   TimeoutError: raised if a service does not produce a port file establish a
                #       connection without a deadline.
                #   ServiceError: raised by an RPC method returning an error status.
                #   NotImplementedError: raised if an RPC method is accessed before the RPC service
                #       has initialized.
                logging.warning(f"{type(e).__name__} {e} (attempt {attempts})")
        else:
            raise TimeoutError(
                f"Failed to create connection to {endpoint} after "
                f"{time() - start_time:.1f} seconds "
                f"({attempts}/{opts.init_max_attempts} attempts made)"
            )

    def __repr__(self):
        if self.connection is None:
            return f"Closed connection to {self.endpoint}"
        return str(self.endpoint)

    @property
    def closed(self) -> bool:
        """Whether the connection is closed."""
        return self.connection is None

    def close(self):
        if self.closed:
            return
        self.connection.close()
        self.connection = None

    def __del__(self):
        # Don't let the subprocess be orphaned if user forgot to close(), or
        # if an exception was thrown.
        self.close()

    def restart(self):
        """Restart a connection a service. If the service is managed by this
        connection (i.e. it is a local binary), the existing service process
        will be killed and replaced. Else, only the connection to the unmanaged
        service process is replaced.
        """
        if self.connection:
            self.connection.close()
        self._establish_connection()

    def __call__(
        self,
        stub_method: StubMethod,
        request: Request,
        timeout: Optional[float] = None,
    ) -> Reply:
        """Invoke an RPC method on the service and return its response. All
        RPC methods accept a single `request` message, and respond with a
        response message.

        Example usage:

        .. code-block:: python

            connection = CompilerGymServiceConnection("localhost:8080")
            request = compiler_gym.service.proto.GetSpacesRequest()
            reply = connection(connection.stub.GetSpaces, request)

        In the above example, the `GetSpaces` RPC method is invoked on a
        connection, yielding a `GetSpacesReply` message.

        :param stub_method: An RPC method attribute on `CompilerGymServiceStub`.
        :param request: A request message.
        :param timeout: The maximum number of seconds to await a reply. If not
            provided, the default value is
            :code:`ConnectionOpts.rpc_call_max_seconds`.
        :raises ValueError: If the service responds with an error indicating an
            invalid argument.
        :raises NotImplementedError: If the service responds with an error
            indicating that the requested functionality is not implemented.
        :raises FileNotFoundError: If the service responds with an error
            indicating that a requested resource was not found.
        :raises OSError: If the service responds with an error indicating that
            it ran out of resources.
        :raises TypeError: If the provided `request` parameter is of
            incorrect type or cannot be serialized, or if the service responds
            with an error indicating that a precondition failed.
        :raises TimeoutError: If the service failed to respond to the query
            within the specified `timeout`.
        :raises ServiceTransportError: If the client failed to communicate with
            the service.
        :raises ServiceIsClosed: If the connection to the service is closed.
        :raises ServiceError: If the service raised an error not covered by
            any of the above conditions.
        :return: A reply message.
        """
        if self.closed:
            self._establish_connection()
        return self.connection(
            stub_method,
            request,
            timeout=timeout or self.opts.rpc_call_max_seconds,
        )
