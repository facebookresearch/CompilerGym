#! /usr/bin/env python3
#
#  Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""An example CompilerGym service in python."""
import os
import sys
from concurrent import futures
from multiprocessing import cpu_count
from pathlib import Path
from signal import SIGTERM, signal
from tempfile import mkdtemp
from threading import Event, Thread
from typing import Type

import grpc
from absl import app, flags, logging

from compiler_gym.service.compilation_session import CompilationSession
from compiler_gym.service.proto import compiler_gym_service_pb2_grpc
from compiler_gym.service.runtime.compiler_gym_service import CompilerGymService
from compiler_gym.util import debug_util as dbg
from compiler_gym.util.filesystem import atomic_file_write
from compiler_gym.util.shell_format import plural

flags.DEFINE_string("working_dir", "", "Path to use as service working directory")
flags.DEFINE_integer("port", 0, "The service listening port")
flags.DEFINE_integer(
    "rpc_service_threads", cpu_count(), "The number of server worker threads"
)
flags.DEFINE_integer("logbuflevel", 0, "Flag for compatability with C++ service.")
FLAGS = flags.FLAGS

MAX_MESSAGE_SIZE_IN_BYTES = 512 * 1024 * 1024


shutdown_signal = Event()


# NOTE(cummins): This script is executed in a subprocess, so code coverage
# tracking does not work. As such we use "# pragma: no cover" annotation for all
# functions.
def _shutdown_handler(signal_number, stack_frame):  # pragma: no cover
    del stack_frame  # Unused
    logging.info("Service received signal: %d", signal_number)
    shutdown_signal.set()


def create_and_run_compiler_gym_service(
    compilation_session_type: Type[CompilationSession],
):  # pragma: no cover
    """Create and run an RPC service for the given compilation session.

    This should be called on its own in a self contained script to implement a
    compilation service. Example:

    .. code-block:: python

        from compiler_gym.service import runtime
        from my_compiler_service import MyCompilationSession

        if __name__ == "__main__":
            runtime.create_and_run_compiler_gym_service(MyCompilationSession)

    This function never returns.

    :param compilation_session_type: A sublass of :class:`CompilationSession
        <compiler_gym.service.CompilationSession>` that provides implementations
        of the abstract methods.
    """

    def main(argv):
        # Register a signal handler for SIGTERM that will set the shutdownSignal
        # future value.
        signal(SIGTERM, _shutdown_handler)

        argv = [x for x in argv if x.strip()]
        if len(argv) > 1:
            print(
                f"ERROR: Unrecognized command line argument '{argv[1]}'",
                file=sys.stderr,
            )
            sys.exit(1)

        working_dir = Path(FLAGS.working_dir or mkdtemp(prefix="compiler_gym-service-"))
        (working_dir / "logs").mkdir(exist_ok=True, parents=True)

        FLAGS.log_dir = str(working_dir / "logs")
        logging.get_absl_handler().use_absl_log_file()
        logging.set_verbosity(dbg.get_logging_level())

        # Create the service.
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=FLAGS.rpc_service_threads),
            options=[
                ("grpc.max_send_message_length", MAX_MESSAGE_SIZE_IN_BYTES),
                ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE_IN_BYTES),
            ],
        )
        service = CompilerGymService(
            working_directory=working_dir,
            compilation_session_type=compilation_session_type,
        )
        compiler_gym_service_pb2_grpc.add_CompilerGymServiceServicer_to_server(
            service, server
        )

        address = f"0.0.0.0:{FLAGS.port}" if FLAGS.port else "0.0.0.0:0"
        port = server.add_insecure_port(address)

        with atomic_file_write(working_dir / "port.txt", fileobj=True, mode="w") as f:
            f.write(str(port))

        with atomic_file_write(working_dir / "pid.txt", fileobj=True, mode="w") as f:
            f.write(str(os.getpid()))

        logging.info(
            "Service %s listening on %d, PID = %d", working_dir, port, os.getpid()
        )

        server.start()

        # Block on the RPC service in a separate thread. This enables the
        # current thread to handle the shutdown routine.
        server_thread = Thread(target=server.wait_for_termination)
        server_thread.start()

        # Block until the shutdown signal is received.
        shutdown_signal.wait()
        logging.info("Shutting down the RPC service")
        server.stop(60).wait()
        server_thread.join()

        if len(service.sessions):
            print(
                "ERROR: Killing a service with",
                plural(len(service.session), "active session", "active sessions"),
                file=sys.stderr,
            )
            sys.exit(6)

    app.run(main)
