# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module contains the logic for connecting to services."""
import logging
import os
import subprocess
import sys
from pathlib import Path
from signal import Signals
from time import sleep, time
from typing import Dict, Iterable, List, Optional, TypeVar, Union

import grpc
from deprecated.sphinx import deprecated
from pydantic import BaseModel

import compiler_gym.errors
from compiler_gym.service.proto import (
    ActionSpace,
    CompilerGymServiceStub,
    GetSpacesReply,
    GetSpacesRequest,
    ObservationSpace,
)
from compiler_gym.service.service_cache import ServiceCache
from compiler_gym.util.debug_util import get_debug_level, logging_level_to_debug_level
from compiler_gym.util.runfiles_path import runfiles_path, site_data_path
from compiler_gym.util.shell_format import join_cmd, plural
from compiler_gym.util.truncate import truncate_lines

GRPC_CHANNEL_OPTIONS = [
    # Disable the inbound message length filter to allow for large messages such
    # as observations.
    ("grpc.max_send_message_length", -1),
    ("grpc.max_receive_message_length", -1),
    # Fix for "received initial metadata size exceeds limit"
    ("grpc.max_metadata_size", 512 * 1024),
    # Spurious error UNAVAILABLE "Trying to connect an http1.x server".
    # https://putridparrot.com/blog/the-unavailable-trying-to-connect-an-http1-x-server-grpc-error/
    ("grpc.enable_http_proxy", 0),
    # Disable TCP port re-use to mitigate port conflict errors when starting
    # many services in parallel. Context:
    # https://github.com/facebookresearch/CompilerGym/issues/572
    ("grpc.so_reuseport", 0),
]

logger = logging.getLogger(__name__)


class ConnectionOpts(BaseModel):
    """The options used to configure a connection to a service."""

    rpc_max_retries: int = 5
    """The maximum number of failed attempts to communicate with the RPC service
    before raising an error. Retries are made only for communication errors.
    Failures from other causes such as error signals raised by the service are
    not retried."""

    retry_wait_seconds: float = 0.1
    """The number of seconds to wait between successive attempts to communicate
    with the RPC service."""

    retry_wait_backoff_exponent: float = 1.5
    """The exponential backoff scaling between successive attempts to
    communicate with the RPC service."""

    init_max_seconds: float = 30
    """The maximum number of seconds to spend attempting to establish a
    connection to the service before failing.
    """

    init_max_attempts: int = 5
    """The maximum number of attempts to make to establish a connection to the
    service before failing.
    """

    local_service_port_init_max_seconds: float = 30
    """The maximum number of seconds to wait for a local service to write the port.txt file."""

    local_service_exit_max_seconds: float = 30
    """The maximum number of seconds to wait for a local service to terminate on close."""

    rpc_init_max_seconds: float = 3
    """The maximum number of seconds to wait for an RPC connection to establish."""

    always_send_benchmark_on_reset: bool = False
    """Send the full benchmark program data to the compiler service on ever call
    to :meth:`env.reset() <compiler_gym.envs.CompilerEnv.reset>`. This is more
    efficient in cases where the majority of calls to
    :meth:`env.reset() <compiler_gym.envs.CompilerEnv.reset>` uses a different
    benchmark. In case of benchmark re-use, leave this :code:`False`.
    """

    script_args: List[str] = []
    """If the service is started from a local script, this set of args is used
    on the command line. No effect when used for existing sockets."""

    script_env: Dict[str, str] = {}
    """If the service is started from a local script, this set of env vars is
    used on the command line. No effect when used for existing sockets."""


# Deprecated since v0.2.4.
# This type is for backwards compatibility that will be removed in a future release.
# Please, use errors from `compiler_gym.errors`.
ServiceError = compiler_gym.errors.ServiceError

# Deprecated since v0.2.4.
# This type is for backwards compatibility that will be removed in a future release.
# Please, use errors from `compiler_gym.errors`.
SessionNotFound = compiler_gym.errors.SessionNotFound

# Deprecated since v0.2.4.
# This type is for backwards compatibility that will be removed in a future release.
# Please, use errors from `compiler_gym.errors`.
ServiceOSError = compiler_gym.errors.ServiceOSError

# Deprecated since v0.2.4.
# This type is for backwards compatibility that will be removed in a future release.
# Please, use errors from `compiler_gym.errors`.
ServiceInitError = compiler_gym.errors.ServiceInitError

# Deprecated since v0.2.4.
# This type is for backwards compatibility that will be removed in a future release.
# Please, use errors from `compiler_gym.errors`.
EnvironmentNotSupported = compiler_gym.errors.EnvironmentNotSupported

# Deprecated since v0.2.4.
# This type is for backwards compatibility that will be removed in a future release.
# Please, use errors from `compiler_gym.errors`.
ServiceTransportError = compiler_gym.errors.ServiceTransportError

# Deprecated since v0.2.4.
# This type is for backwards compatibility that will be removed in a future release.
# Please, use errors from `compiler_gym.errors`.
ServiceIsClosed = compiler_gym.errors.ServiceIsClosed

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

        def __call__(
            self, a: Request, timeout: float
        ) -> Reply:  # pylint: disable=undefined-variable
            ...

else:
    # Legacy support for Python < 3.8.
    from typing import Callable

    StubMethod = Callable[[Request], Reply]


class Connection:
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
        max_retries=5,
        retry_wait_seconds=0.1,
        retry_wait_backoff_exponent=1.5,
    ) -> Reply:
        """Call the service with the given arguments."""
        # pylint: disable=no-member
        #
        # House keeping note: if you modify the exceptions that this method
        # raises, please update the CompilerGymServiceConnection.__call__()
        # docstring.
        attempt = 0
        while True:
            try:
                return stub_method(request, timeout=timeout)
            except ValueError as e:
                if str(e) == "Cannot invoke RPC on closed channel!":
                    raise ServiceIsClosed(
                        "RPC communication failed because channel is closed"
                    ) from None
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
                    raise ServiceOSError(e.details()) from None
                elif e.code() == grpc.StatusCode.FAILED_PRECONDITION:
                    raise TypeError(str(e.details())) from None
                elif e.code() == grpc.StatusCode.UNAVAILABLE:
                    # For "unavailable" errors we retry with exponential
                    # backoff. This is because this error can be caused by an
                    # overloaded service, a flaky connection, etc.

                    # Early exit in case we can detect that the service is down
                    # and so there is no use in retrying the RPC call.
                    if self.service_is_down():
                        raise ServiceIsClosed("Service is offline")

                    attempt += 1
                    if attempt > max_retries:
                        raise ServiceTransportError(
                            f"{self.url} {e.details()} ({max_retries} retries)"
                        ) from None
                    remaining = max_retries - attempt
                    logger.warning(
                        "%s %s (%d %s remaining)",
                        self.url,
                        e.details(),
                        remaining,
                        plural(remaining, "attempt", "attempts"),
                    )
                    sleep(retry_wait_seconds)
                    retry_wait_seconds *= retry_wait_backoff_exponent
                elif (
                    e.code() == grpc.StatusCode.INTERNAL
                    and e.details() == "Exception serializing request!"
                ):
                    raise TypeError(
                        f"{e.details()} Request type: {type(request).__name__}"
                    ) from None
                elif e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    raise TimeoutError(
                        f"{e.details()} ({timeout:.1f} seconds)"
                    ) from None
                elif e.code() == grpc.StatusCode.DATA_LOSS:
                    raise ServiceError(e.details()) from None
                elif e.code() == grpc.StatusCode.UNKNOWN:
                    # By default, GRPC provides no context if an exception is
                    # raised in an RPC handler as this could lead to an
                    # information leak. Unfortunately for us this makes
                    # debugging a little more difficult, so be verbose about the
                    # possible causes of this error.
                    raise ServiceError(
                        "Service returned an unknown error. Possibly an "
                        "unhandled exception in a C++ RPC handler, see "
                        "<https://github.com/grpc/grpc/issues/13706>."
                    ) from None
                else:
                    raise ServiceError(
                        f"RPC call returned status code {e.code()} and error `{e.details()}`"
                    ) from None

    def loglines(self) -> Iterable[str]:
        """Fetch any available log lines from the service backend.

        :return: An iterator over lines of logs.
        """
        yield from ()

    def service_is_down(self) -> bool:
        """Return true if the service is known to be dead.

        Subclasses can use this for fast checks that a service is down to avoid
        retry loops.
        """
        return False


class ManagedConnection(Connection):
    """A connection to a service using a managed subprocess."""

    def __init__(
        self,
        local_service_binary: Path,
        port_init_max_seconds: float,
        rpc_init_max_seconds: float,
        process_exit_max_seconds: float,
        script_args: List[str],
        script_env: Dict[str, str],
    ):
        """Constructor.

        :param local_service_binary: The path of the service binary.
        :raises TimeoutError: If fails to establish connection within a specified time limit.
        """
        self.process_exit_max_seconds = process_exit_max_seconds

        if not Path(local_service_binary).is_file():
            raise FileNotFoundError(f"File not found: {local_service_binary}")
        self.cache = ServiceCache()

        # The command that will be executed. The working directory of this
        # command will be set to the local_service_binary's parent, so we can
        # use the relpath for a neater `ps aux` view.
        cmd = [
            f"./{local_service_binary.name}",
            f"--working_dir={self.cache.path}",
        ]
        # Add any custom arguments
        cmd += script_args

        # Set the root of the runfiles directory.
        env = os.environ.copy()
        env["COMPILER_GYM_RUNFILES"] = str(runfiles_path("."))
        env["COMPILER_GYM_SITE_DATA"] = str(site_data_path("."))
        # Set the pythonpath so that executable python scripts can use absolute
        # import paths like `from compiler_gym.envs.foo import bar`.
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f'{env["PYTHONPATH"]}:{env["COMPILER_GYM_RUNFILES"]}'
        else:
            env["PYTHONPATH"] = env["COMPILER_GYM_RUNFILES"]

        # Set the verbosity of the service. The logging level of the service is
        # the debug level - 1, so that COMPILER_GYM_DEBUG=3 will cause VLOG(2)
        # and lower to be logged to stdout.
        debug_level = max(
            get_debug_level(), logging_level_to_debug_level(logger.getEffectiveLevel())
        )
        if debug_level > 0:
            cmd.append("--alsologtostderr")
            cmd.append(f"-v={debug_level - 1}")
            # If we are debugging the backend, set the logbuflevel to a low
            # value to disable buffering of logging messages. This removes any
            # buffering between `LOG(INFO) << "..."` and the message being
            # emited to stderr.
            cmd.append("--logbuflevel=-1")
        else:
            # Silence the gRPC logs as we will do our own error reporting, but
            # don't override any existing value so that the user may debug the
            # gRPC backend by setting GRPC_VERBOSITY to ERROR, INFO, or DEBUG.
            if not os.environ.get("GRPC_VERBOSITY"):
                env["GRPC_VERBOSITY"] = "NONE"

        # Set environment variable COMPILER_GYM_SERVICE_ARGS to pass
        # additional arguments to the service.
        args = os.environ.get("COMPILER_GYM_SERVICE_ARGS", "")
        if args:
            cmd.append(args)

        # Add any custom environment variables
        env.update(script_env)

        logger.debug(
            "Exec `%s%s`",
            " ".join(f"{k}={v}" for k, v in script_env.items()) + " "
            if script_env
            else "",
            join_cmd(cmd),
        )

        self.process = subprocess.Popen(
            cmd,
            env=env,
            cwd=local_service_binary.parent,
        )
        self._process_returncode_exception_raised = False

        # Read the port from a file generated by the service.
        wait_secs = 0.1
        port_path = self.cache / "port.txt"
        end_time = time() + port_init_max_seconds
        while time() < end_time:
            returncode = self.process.poll()
            if returncode is not None:
                try:
                    # Try and decode the name of a signal. Signal returncodes
                    # are negative.
                    returncode = f"{returncode} ({Signals(abs(returncode)).name})"
                except ValueError:
                    pass
                msg = f"Service terminated with returncode: {returncode}"
                # Attach any logs from the service if available.
                logs = truncate_lines(
                    self.loglines(), max_line_len=100, max_lines=25, tail=True
                )
                if logs:
                    msg = f"{msg}\nService logs:\n{logs}"
                self.cache.close()
                raise ServiceError(msg)
            if port_path.is_file():
                try:
                    with open(port_path) as f:
                        self.port = int(f.read().rstrip())
                    break
                except ValueError:
                    # ValueError is raised by int(...) on invalid input. In that
                    # case, wait for longer.
                    pass
            sleep(wait_secs)
            wait_secs *= 1.2
        else:
            # kill() was added in Python 3.7.
            if sys.version_info >= (3, 7, 0):
                self.process.kill()
            else:
                self.process.terminate()
            self.process.communicate(timeout=rpc_init_max_seconds)
            self.cache.close()
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
                logger.debug(
                    "Connection attempt %d = %s %s", attempts, type(e).__name__, str(e)
                )
                wait_secs *= 1.2
        else:
            # kill() was added in Python 3.7.
            if sys.version_info >= (3, 7, 0):
                self.process.kill()
            else:
                self.process.terminate()
            self.process.communicate(timeout=process_exit_max_seconds)

            # Include the last few lines of logs generated by the compiler
            # service, if any.
            logs = truncate_lines(
                self.loglines(), max_line_len=100, max_lines=25, tail=True
            )
            logs_message = f" Service logs:\n{logs}" if logs else ""

            self.cache.close()
            raise TimeoutError(
                "Failed to connect to RPC service after "
                f"{rpc_init_max_seconds:.1f} seconds.{logs_message}"
            )

        super().__init__(channel, url)

    @property
    @deprecated(version="0.2.4", reason="Replace `working_directory` with `cache.path`")
    def working_dir(self) -> Path:
        return self.cache.path

    def service_is_down(self) -> bool:
        """Return true if the service subprocess has terminated."""
        return self.process.poll() is not None

    def loglines(self) -> Iterable[str]:
        """Fetch any available log lines from the service backend.

        :return: An iterator over lines of logs.
        """
        # Compiler services write log files in the logs directory. Iterate over
        # them and return their contents.
        if not (self.cache / "logs").is_dir():
            return ()
        for path in sorted((self.cache / "logs").iterdir()):
            if not path.is_file():
                continue
            with open(path) as f:
                yield from f.readlines()

    def close(self):
        """Terminate a local subprocess and close the connection."""
        try:
            self.process.terminate()
            self.process.communicate(timeout=self.process_exit_max_seconds)
            if (
                self.process.returncode
                and not self._process_returncode_exception_raised
            ):
                # You can call close() multiple times but we only want to emit
                # the exception once.
                self._process_returncode_exception_raised = True
                raise ServiceError(
                    f"Service exited with returncode {self.process.returncode}"
                )
        except ServiceIsClosed:
            # The service has already been closed, nothing to do.
            pass
        except ProcessLookupError:
            logger.warning("Service process not found at %s", self.cache)
        except subprocess.TimeoutExpired:
            # Try and kill it and then walk away.
            try:
                # kill() was added in Python 3.7.
                if sys.version_info >= (3, 7, 0):
                    self.process.kill()
                else:
                    self.process.terminate()
                self.process.communicate(timeout=60)
            except:  # noqa
                pass
            logger.warning("Abandoning orphan service at %s", self.cache)
        finally:
            self.cache.close()
            super().close()

    def __repr__(self):
        if self.process.poll() is None:
            return (
                f"Connection to service at {self.url} running on PID {self.process.pid}"
            )
        return f"Connection to dead service at {self.url}"


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
                logger.debug(
                    "Connection attempt %d = %s %s", attempts, type(e).__name__, str(e)
                )
                wait_secs *= 1.2
        else:
            raise TimeoutError(
                f"Failed to connect to {url} after "
                f"{rpc_init_max_seconds:.1f} seconds"
            )

        super().__init__(channel, url)

    def __repr__(self):
        return f"Connection to unmanaged service {self.url}"


class CompilerGymServiceConnection:
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
        connection(connection.stub.StartSession, StartSessionRequest())
        # Close the connection. The service running on port 8080 is
        # left running.
        connection.close()

    Example usage of a managed service connection:

    .. code-block:: python

        # Start a subprocess using the binary located at /path/to/my/service.
        connection = CompilerGymServiceConnection(Path("/path/to/my/service"))
        # Invoke an RPC method.
        connection(connection.stub.StartSession, StartSessionRequest())
        # Close the connection. The subprocess is terminated.
        connection.close()

    :ivar stub: A CompilerGymServiceStub that can be used as the first argument
        to :py:meth:`__call__()` to specify an RPC
        method to call.
    :ivar action_spaces: A list of action spaces provided by the service.
    :ivar observation_spaces: A list of observation spaces provided by the
        service.
    """

    def __init__(
        self,
        endpoint: Union[str, Path],
        opts: ConnectionOpts = None,
    ):
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

    def _establish_connection(self) -> None:
        """Create and establish a connection."""
        self.connection = self._create_connection(self.endpoint, self.opts)
        self.stub = self.connection.stub

    @classmethod
    def _create_connection(
        cls,
        endpoint: Union[str, Path],
        opts: ConnectionOpts,
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
        last_exception = None
        while time() < end_time and attempts < opts.init_max_attempts:
            attempts += 1
            try:
                if isinstance(endpoint, Path):
                    endpoint_name = endpoint.name
                    return ManagedConnection(
                        local_service_binary=endpoint,
                        process_exit_max_seconds=opts.local_service_exit_max_seconds,
                        rpc_init_max_seconds=opts.rpc_init_max_seconds,
                        port_init_max_seconds=opts.local_service_port_init_max_seconds,
                        script_args=opts.script_args,
                        script_env=opts.script_env,
                    )
                else:
                    endpoint_name = endpoint
                    return UnmanagedConnection(
                        url=endpoint,
                        rpc_init_max_seconds=opts.rpc_init_max_seconds,
                    )
            except (TimeoutError, ServiceError, NotImplementedError) as e:
                # Catch preventable errors so that we can retry:
                #   TimeoutError: raised if a service does not produce a port file establish a
                #       connection without a deadline.
                #   ServiceError: raised by an RPC method returning an error status.
                #   NotImplementedError: raised if an RPC method is accessed before the RPC service
                #       has initialized.
                last_exception = e
                logger.warning("%s %s (attempt %d)", type(e).__name__, e, attempts)

        exception_class = (
            ServiceError if attempts >= opts.init_max_attempts else TimeoutError
        )
        raise exception_class(
            f"Failed to create connection to {endpoint_name} after "
            f"{time() - start_time:.1f} seconds "
            f"({attempts} {plural(attempts, 'attempt', 'attempts')} made).\n"
            f"Last error ({type(last_exception).__name__}): {last_exception}"
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
        timeout: Optional[float] = 300,
        max_retries: Optional[int] = None,
        retry_wait_seconds: Optional[float] = None,
        retry_wait_backoff_exponent: Optional[float] = None,
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
        :param timeout: The maximum number of seconds to await a reply.
        :param max_retries: The maximum number of failed attempts to communicate
            with the RPC service before raising an error. Retries are made only
            for communication errors. Failures from other causes such as error
            signals raised by the service are not retried.
        :param retry_wait_seconds: The number of seconds to wait between
            successive attempts to communicate with the RPC service.
        :param retry_wait_backoff_exponent: The exponential backoff scaling
            between successive attempts to communicate with the RPC service.
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
            timeout=timeout,
            max_retries=max_retries or self.opts.rpc_max_retries,
            retry_wait_seconds=retry_wait_seconds or self.opts.retry_wait_seconds,
            retry_wait_backoff_exponent=(
                retry_wait_backoff_exponent or self.opts.retry_wait_backoff_exponent
            ),
        )
