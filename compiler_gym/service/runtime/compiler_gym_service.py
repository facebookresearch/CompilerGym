# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import traceback
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Dict, Optional

from grpc import StatusCode

from compiler_gym.service.compilation_session import CompilationSession
from compiler_gym.service.proto import AddBenchmarkReply, AddBenchmarkRequest
from compiler_gym.service.proto import (
    CompilerGymServiceServicer as CompilerGymServiceServicerStub,
)
from compiler_gym.service.proto import (
    EndSessionReply,
    EndSessionRequest,
    ForkSessionReply,
    ForkSessionRequest,
    GetSpacesReply,
    GetSpacesRequest,
    GetVersionReply,
    GetVersionRequest,
    SendSessionParameterReply,
    SendSessionParameterRequest,
    StartSessionReply,
    StartSessionRequest,
    StepReply,
    StepRequest,
)
from compiler_gym.service.runtime.benchmark_cache import BenchmarkCache
from compiler_gym.util.version import __version__

logger = logging.getLogger(__name__)

# NOTE(cummins): The CompilerGymService class is used in a subprocess by a
# compiler service, so code coverage tracking does not work. As such we use "#
# pragma: no cover" annotation for all definitions in this file.


@contextmanager
def exception_to_grpc_status(context):  # pragma: no cover
    def handle_exception_as(exception, code):
        exception_trace = "".join(
            traceback.TracebackException.from_exception(exception).format()
        )
        logger.warning("%s", exception_trace)
        context.set_code(code)
        context.set_details(str(exception))

    try:
        yield
    except ValueError as e:
        handle_exception_as(e, StatusCode.INVALID_ARGUMENT)
    except LookupError as e:
        handle_exception_as(e, StatusCode.NOT_FOUND)
    except NotImplementedError as e:
        handle_exception_as(e, StatusCode.UNIMPLEMENTED)
    except FileNotFoundError as e:
        handle_exception_as(e, StatusCode.UNIMPLEMENTED)
    except TypeError as e:
        handle_exception_as(e, StatusCode.FAILED_PRECONDITION)
    except TimeoutError as e:
        handle_exception_as(e, StatusCode.DEADLINE_EXCEEDED)
    except Exception as e:  # pylint: disable=broad-except
        handle_exception_as(e, StatusCode.INTERNAL)


class CompilerGymService(CompilerGymServiceServicerStub):  # pragma: no cover
    def __init__(self, working_directory: Path, compilation_session_type):
        """Constructor.

        :param working_directory: The working directory for this service.

        :param compilation_session_type: The :class:`CompilationSession
            <compiler_gym.service.CompilationSession>` type that this service
            implements.
        """
        self.working_directory = working_directory
        self.benchmarks = BenchmarkCache()

        self.compilation_session_type = compilation_session_type
        self.sessions: Dict[int, CompilationSession] = {}
        self.sessions_lock = Lock()
        self.next_session_id: int = 0

        self.action_spaces = compilation_session_type.action_spaces
        self.observation_spaces = compilation_session_type.observation_spaces

    def GetVersion(self, request: GetVersionRequest, context) -> GetVersionReply:
        del context  # Unused
        del request  # Unused
        logger.debug("GetVersion()")
        return GetVersionReply(
            service_version=__version__,
            compiler_version=self.compilation_session_type.compiler_version,
        )

    def GetSpaces(self, request: GetSpacesRequest, context) -> GetSpacesReply:
        del request  # Unused
        logger.debug("GetSpaces()")
        with exception_to_grpc_status(context):
            return GetSpacesReply(
                action_space_list=self.action_spaces,
                observation_space_list=self.observation_spaces,
            )

    def StartSession(self, request: StartSessionRequest, context) -> StartSessionReply:
        """Create a new compilation session."""
        logger.debug(
            "StartSession(id=%d, benchmark=%s), %d active sessions",
            self.next_session_id,
            request.benchmark.uri,
            len(self.sessions) + 1,
        )
        reply = StartSessionReply()

        if not request.benchmark:
            context.set_code(StatusCode.INVALID_ARGUMENT)
            context.set_details("No benchmark URI set for StartSession()")
            return reply

        with self.sessions_lock, exception_to_grpc_status(context):
            # If a benchmark definition was provided, add it.
            if request.benchmark.HasField("program"):
                self.benchmarks[request.benchmark.uri] = request.benchmark

            # Lookup the requested benchmark.
            if request.benchmark.uri not in self.benchmarks:
                context.set_code(StatusCode.NOT_FOUND)
                context.set_details("Benchmark not found")
                return reply

            session = self.compilation_session_type(
                working_directory=self.working_directory,
                action_space=self.action_spaces[request.action_space],
                benchmark=self.benchmarks[request.benchmark.uri],
            )

            # Generate the initial observations.
            reply.observation.extend(
                [
                    session.get_observation(self.observation_spaces[obs])
                    for obs in request.observation_space
                ]
            )

            reply.session_id = self.next_session_id
            self.sessions[reply.session_id] = session
            self.next_session_id += 1

        return reply

    def ForkSession(self, request: ForkSessionRequest, context) -> ForkSessionReply:
        logger.debug(
            "ForkSession(id=%d), [%s]",
            request.session_id,
            self.next_session_id,
        )

        reply = ForkSessionReply()
        with exception_to_grpc_status(context):
            session = self.sessions[request.session_id]
            self.sessions[reply.session_id] = session.fork()
            reply.session_id = self.next_session_id
            self.next_session_id += 1

        return reply

    def EndSession(self, request: EndSessionRequest, context) -> EndSessionReply:
        del context  # Unused
        logger.debug(
            "EndSession(id=%d), %d sessions remaining",
            request.session_id,
            len(self.sessions) - 1,
        )

        with self.sessions_lock:
            if request.session_id in self.sessions:
                del self.sessions[request.session_id]
            return EndSessionReply(remaining_sessions=len(self.sessions))

    def Step(self, request: StepRequest, context) -> StepReply:
        logger.debug("Step()")
        reply = StepReply()

        if request.session_id not in self.sessions:
            context.set_code(StatusCode.NOT_FOUND)
            context.set_details(f"Session not found: {request.session_id}")
            return reply

        reply.action_had_no_effect = True

        with exception_to_grpc_status(context):
            session = self.sessions[request.session_id]
            for action in request.action:
                reply.end_of_session, nas, ahne = session.apply_action(action)
                reply.action_had_no_effect &= ahne
                if nas:
                    reply.new_action_space.CopyFrom(nas)

            reply.observation.extend(
                [
                    session.get_observation(self.observation_spaces[obs])
                    for obs in request.observation_space
                ]
            )

        return reply

    def AddBenchmark(self, request: AddBenchmarkRequest, context) -> AddBenchmarkReply:
        del context  # Unused
        reply = AddBenchmarkReply()
        with self.sessions_lock:
            for benchmark in request.benchmark:
                self.benchmarks[benchmark.uri] = benchmark
        return reply

    def SendSessionParameter(
        self, request: SendSessionParameterRequest, context
    ) -> SendSessionParameterReply:
        reply = SendSessionParameterReply()

        if request.session_id not in self.sessions:
            context.set_code(StatusCode.NOT_FOUND)
            context.set_details(f"Session not found: {request.session_id}")
            return reply

        session = self.sessions[request.session_id]

        with exception_to_grpc_status(context):
            for param in request.parameter:
                # Handle each parameter in the session and generate a response.
                message = session.handle_session_parameter(param.key, param.value)

                # Use the builtin parameter handlers if not handled by a session.
                if message is None:
                    message = self._handle_builtin_session_parameter(
                        param.key, param.value
                    )

                if message is None:
                    context.set_code(StatusCode.INVALID_ARGUMENT)
                    context.set_details(f"Unknown parameter: {param.key}")
                    return reply
                reply.reply.append(message)

        return reply

    def _handle_builtin_session_parameter(self, key: str, value: str) -> Optional[str]:
        """Handle a built-in session parameter.

        :param key: The parameter key.

        :param value: The parameter value.

        :return: The response message, or :code:`None` if the key is not
            understood.
        """
        if key == "service.benchmark_cache.set_max_size_in_bytes":
            self.benchmarks.set_max_size_in_bytes = int(value)
            return value
        elif key == "service.benchmark_cache.get_max_size_in_bytes":
            return str(self.benchmarks.max_size_in_bytes)
        elif key == "service.benchmark_cache.get_size_in_bytes":
            return str(self.benchmarks.size_in_bytes)

        return None
