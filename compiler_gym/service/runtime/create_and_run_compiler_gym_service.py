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
from tempfile import mkdtemp

import grpc
from absl import app, flags, logging

from compiler_gym.service.proto import compiler_gym_service_pb2_grpc
from compiler_gym.service.runtime.compiler_gym_service import CompilerGymService
from compiler_gym.util.filesystem import atomic_file_write

flags.DEFINE_string("working_dir", "", "Path to use as service working directory")
flags.DEFINE_integer("port", 0, "The service listening port")
flags.DEFINE_integer("nproc", cpu_count(), "The number of server worker threads")
flags.DEFINE_integer("logbuflevel", 0, "Flag for compatability with C++ service.")
FLAGS = flags.FLAGS

MAX_MESSAGE_SIZE_IN_BYTES = 512 * 1024 * 1024


def create_and_run_compiler_gym_service(compilation_session_type):
    def main(argv):
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

        # Create the service.
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=FLAGS.nproc),
            options=[
                ("grpc.max_send_message_length", MAX_MESSAGE_SIZE_IN_BYTES),
                ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE_IN_BYTES),
            ],
        )
        servicer = CompilerGymService(
            working_directory=working_dir,
            compilation_session_type=compilation_session_type,
        )
        compiler_gym_service_pb2_grpc.add_CompilerGymServiceServicer_to_server(
            servicer, server
        )
        port = server.add_insecure_port("0.0.0.0:0")

        with atomic_file_write(working_dir / "port.txt", fileobj=True, mode="w") as f:
            f.write(str(port))

        with atomic_file_write(working_dir / "pid.txt", fileobj=True, mode="w") as f:
            f.write(str(os.getpid()))

        logging.info(
            "Service %s listening on %d, PID = %d", working_dir, port, os.getpid()
        )

        server.start()
        server.wait_for_termination()
        logging.fatal(
            "Unreachable! grpc.server.wait_for_termination() should not return"
        )

    app.run(main)
