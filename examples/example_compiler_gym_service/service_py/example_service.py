#! /usr/bin/env python3
#
#  Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""An example CompilerGym service in python."""
import logging
from concurrent import futures
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict

import grpc
from absl import app, flags

import compiler_gym
from compiler_gym.service import proto
from compiler_gym.service.proto import compiler_gym_service_pb2_grpc

flags.DEFINE_string(
    "working_dir",
    "/tmp/example_compiler_gym_service",
    "Path to use as service working directory",
)
flags.DEFINE_integer("port", 0, "The service listening port")
flags.DEFINE_integer("nproc", cpu_count(), "The number of server worker threads")
flags.DEFINE_integer("logbuflevel", 0, "Flag for compatability with C++ service.")
FLAGS = flags.FLAGS

# For development / debugging, set environment variable COMPILER_GYM_DEBUG=3.
# This will cause all output of this script to be logged to stdout. Otherwise
# all output from this process is silently thrown away.
logging.basicConfig(level=logging.DEBUG)

# The names of the benchmarks that are supported
BENCHMARKS = ["benchmark://example-v0/foo", "benchmark://example-v0/bar"]

# The list of actions that are supported by this service. This example uses a
# static (unchanging) action space, but this could be extended to support a
# dynamic action space.
ACTION_SPACE = proto.ActionSpace(
    name="default",
    action=[
        "a",
        "b",
        "c",
    ],
)

# A list of observation spaces supported by this service. Each of these
# ObservationSpace protos describes an observation space.
OBSERVATION_SPACES = [
    proto.ObservationSpace(
        name="ir",
        string_size_range=proto.ScalarRange(min=proto.ScalarLimit(value=0)),
        deterministic=True,
        platform_dependent=False,
        default_value=proto.Observation(string_value=""),
    ),
    proto.ObservationSpace(
        name="features",
        int64_range_list=proto.ScalarRangeList(
            range=[
                proto.ScalarRange(
                    min=proto.ScalarLimit(value=-100), max=proto.ScalarLimit(value=100)
                ),
                proto.ScalarRange(
                    min=proto.ScalarLimit(value=-100), max=proto.ScalarLimit(value=100)
                ),
                proto.ScalarRange(
                    min=proto.ScalarLimit(value=-100), max=proto.ScalarLimit(value=100)
                ),
            ]
        ),
    ),
    proto.ObservationSpace(
        name="runtime",
        scalar_double_range=proto.ScalarRange(min=proto.ScalarLimit(value=0)),
        deterministic=False,
        platform_dependent=True,
        default_value=proto.Observation(
            scalar_double=0,
        ),
    ),
]


class CompilationSession(object):
    """Represents an instance of an interactive compilation session."""

    def __init__(self, benchmark: str):
        # Do any of the set up required to start a compilation "session".
        self.benchmark = benchmark

    def step(self, request: proto.StepRequest, context) -> proto.StepReply:
        reply = proto.StepReply()

        # Apply a list of actions from the user. Each value is an index into the
        # ACTIONS_SPACE.action list.
        for action in request.action:
            logging.debug("Apply action %d", action)
            if action < 0 or action >= len(ACTION_SPACE.action):
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Out-of-range")
                return

        # Compute a list of observations from the user. Each value is an index
        # into the OBSERVATION_SPACES list.
        for observation_space in request.observation_space:
            logging.debug("Compute observation %d", observation_space)
            observation = reply.observation.add()
            if observation_space == 0:  # ir
                observation.string_value = "Hello, world!"
            elif observation_space == 1:  # features
                observation.int64_list.value[:] = [0, 0, 0]
            elif observation_space == 1:  # runtime
                observation.scalar_double = 0

        return reply


class ExampleCompilerGymService(proto.CompilerGymServiceServicer):
    """The service creates and manages sessions, and is responsible for
    reporting the service capabilities to the user."""

    def __init__(self, working_dir: Path):
        self.working_dir = working_dir
        self.sessions: Dict[int, CompilationSession] = {}

    def GetVersion(
        self, request: proto.GetVersionRequest, context
    ) -> proto.GetVersionReply:
        del context  # Unused
        del request  # Unused
        logging.debug("GetVersion()")
        return proto.GetVersionReply(
            service_version=compiler_gym.__version__,
            compiler_version="1.0.0",  # Arbitrary version tag.
        )

    def GetSpaces(
        self, request: proto.GetSpacesRequest, context
    ) -> proto.GetSpacesReply:
        del context  # Unused
        del request  # Unused
        logging.debug("GetSpaces()")
        return proto.GetSpacesReply(
            action_space_list=[ACTION_SPACE],
            observation_space_list=OBSERVATION_SPACES,
        )

    def GetBenchmarks(
        self, request: proto.GetBenchmarksRequest, context
    ) -> proto.GetBenchmarksReply:
        del context  # Unused
        # Report the available benchmarks to the user.
        logging.debug("GetBenchmarks()")
        return proto.GetBenchmarksReply(benchmark=BENCHMARKS)

    def StartSession(
        self, request: proto.StartSessionRequest, context
    ) -> proto.StartSessionReply:
        """Create a new compilation session."""
        logging.debug("StartSession(benchmark=%s)", request.benchmark)

        if not request.benchmark:
            benchmark = "foo"  # Pick a default benchmark is none was requested.
        else:
            benchmark = request.benchmark

        if benchmark not in BENCHMARKS:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Unknown program name")
            return

        session = CompilationSession(benchmark=benchmark)
        session_id = len(self.sessions)
        self.sessions[session_id] = session
        return proto.StartSessionReply(
            session_id=session_id, benchmark=session.benchmark
        )

    def EndSession(
        self, request: proto.EndSessionRequest, context
    ) -> proto.EndSessionReply:
        """End a compilation session."""
        del context  # Unused
        logging.debug("EndSession()")
        if request.session_id in self.sessions:
            del self.sessions[request.session_id]
        return proto.EndSessionReply(remaining_sessions=len(self.sessions))

    def Step(self, request: proto.StepRequest, context) -> proto.StepReply:
        logging.debug("Step()")
        if request.session_id not in self.sessions:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Session ID not found")
            return

        return self.sessions[request.session_id].step(request, context)


def main(argv):
    assert argv  # Unused

    working_dir = Path(FLAGS.working_dir)
    working_dir.mkdir(exist_ok=True, parents=True)

    # Create the service.
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=FLAGS.nproc))
    compiler_gym_service_pb2_grpc.add_CompilerGymServiceServicer_to_server(
        ExampleCompilerGymService(working_dir), server
    )
    port = server.add_insecure_port("0.0.0.0:0")
    logging.info("Starting service on %s with working dir %s", port, working_dir)

    with open(working_dir / "port.txt", "w") as f:
        f.write(str(port))

    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    app.run(main)
