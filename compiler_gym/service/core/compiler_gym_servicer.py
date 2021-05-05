# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Dict

from grpc import StatusCode

from compiler_gym.service.core.core import CompilationSession
from compiler_gym.service.proto import AddBenchmarkReply, AddBenchmarkRequest, Benchmark
from compiler_gym.service.proto import (
    CompilerGymServiceServicer as CompilerGymServiceServicerStub,
)
from compiler_gym.service.proto import (
    EndSessionReply,
    EndSessionRequest,
    GetSpacesReply,
    GetSpacesRequest,
    GetVersionReply,
    GetVersionRequest,
    StartSessionReply,
    StartSessionRequest,
    StepReply,
    StepRequest,
)
from compiler_gym.util.version import __version__


@contextmanager
def exception_to_grpc_status(context):
    def handle_exception_as(exception, code):
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


class CompilerGymServicer(CompilerGymServiceServicerStub):
    def __init__(self, working_directory: Path, compilation_session_type):
        self.working_directory = working_directory
        self.compilation_session_type = compilation_session_type
        self.sessions: Dict[int, CompilationSession] = {}
        self.next_session_id: int = 0
        self.sessions_lock = Lock()
        # TODO: Replace with benchmark cache
        self.benchmark_cache: Dict[str, Benchmark] = {}

    def GetVersion(self, request: GetVersionRequest, context) -> GetVersionReply:
        del context  # Unused
        del request  # Unused
        logging.debug("GetVersion()")
        return GetVersionReply(
            service_version=__version__,
            compiler_version=self.compilation_session_type.compiler_version,
        )

    def GetSpaces(self, request: GetSpacesRequest, context) -> GetSpacesReply:
        del request  # Unused
        logging.debug("GetSpaces()")
        with exception_to_grpc_status(context):
            return GetSpacesReply(
                action_space_list=self.compilation_session_type.action_spaces,
                observation_space_list=self.compilation_session_type.observation_spaces,
            )
        return GetSpacesReply()

    def StartSession(self, request: StartSessionRequest, context) -> StartSessionReply:
        """Create a new compilation session."""
        logging.debug(
            "StartSession(benchmark=%s), [%d]", request.benchmark, self.next_session_id
        )
        reply = StartSessionReply()

        if not request.benchmark:
            context.set_code(StatusCode.INVALID_ARGUMENT)
            context.set_details("No benchmark URI set for StartSession()")
            return reply

        with self.sessions_lock:
            if request.benchmark not in self.benchmark_cache:
                context.set_code(StatusCode.NOT_FOUND)
                context.set_details("Benchmark not found")
                return reply

            with exception_to_grpc_status(context):
                session = self.compilation_session_type(
                    working_directory=self.working_directory,
                    action_space_index=request.action_space,
                    benchmark=self.benchmark_cache[request.benchmark],
                )

                # Generate the initial observations.
                reply.observation.extend(
                    [session.get_observation(obs) for obs in request.observation_space]
                )

                reply.session_id = self.next_session_id
                self.sessions[reply.session_id] = session
                self.next_session_id += 1

            return reply

    def EndSession(self, request: EndSessionRequest, context) -> EndSessionReply:
        del context  # Unused
        logging.debug(
            "EndSession(%d), %d sessions remaining",
            request.session_id,
            len(self.sessions) - 1,
        )

        with self.sessions_lock:
            if request.session_id in self.sessions:
                del self.sessions[request.session_id]
            return EndSessionReply(remaining_sessions=len(self.sessions))

    def Step(self, request: StepRequest, context) -> StepReply:
        logging.debug("Step()")
        reply = StepReply()

        if request.session_id not in self.sessions:
            context.set_code(StatusCode.NOT_FOUND)
            context.set_details(f"Session not found: {request.session_id}")
            return reply

        session = self.sessions[request.session_id]

        with exception_to_grpc_status(context):
            (
                reply.end_of_session,
                new_action_space,
                reply.action_had_no_effect,
            ) = session.apply_actions(request.action)
            if new_action_space:
                reply.new_action_space.CopyFrom(new_action_space)

            reply.observation.extend(
                [session.get_observation(obs) for obs in request.observation_space]
            )

        return reply

    def AddBenchmark(self, request: AddBenchmarkRequest, context) -> AddBenchmarkReply:
        del context  # Unused
        reply = AddBenchmarkReply()
        with self.sessions_lock:
            for benchmark in request.benchmark:
                self.benchmark_cache[benchmark.uri] = benchmark
        return reply
