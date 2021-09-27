#! /usr/bin/env python3
#
#  Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""An example CompilerGym service in python."""
import logging
import subprocess
import sys
from pathlib import Path
from signal import Signals
from typing import List, Optional, Tuple

from compiler_gym.service import CompilationSession
from compiler_gym.service.proto import (
    Action,
    ActionSpace,
    Benchmark,
    BenchmarkInitError,
    Observation,
    ObservationSpace,
    ScalarLimit,
    ScalarRange,
    ScalarRangeList,
)
from compiler_gym.service.runtime import create_and_run_compiler_gym_service


class UnrollingCompilationSession(CompilationSession):
    """Represents an instance of an interactive compilation session."""

    compiler_version: str = "1.0.0"

    # The list of actions that are supported by this service. This example uses
    # a static (unchanging) action space, but this could be extended to support
    # a dynamic action space.
    action_spaces = [
        ActionSpace(
            name="default",
            action=[
                "a",
                "b",
                "c",
            ],
        )
    ]

    # A list of observation spaces supported by this service. Each of these
    # ObservationSpace protos describes an observation space.
    observation_spaces = [
        ObservationSpace(
            name="ir",
            string_size_range=ScalarRange(min=ScalarLimit(value=0)),
            deterministic=True,
            platform_dependent=False,
            default_value=Observation(string_value=""),
        ),
        ObservationSpace(
            name="features",
            int64_range_list=ScalarRangeList(
                range=[
                    ScalarRange(
                        min=ScalarLimit(value=-100), max=ScalarLimit(value=100)
                    ),
                    ScalarRange(
                        min=ScalarLimit(value=-100), max=ScalarLimit(value=100)
                    ),
                    ScalarRange(
                        min=ScalarLimit(value=-100), max=ScalarLimit(value=100)
                    ),
                ]
            ),
        ),
        ObservationSpace(
            name="runtime",
            scalar_double_range=ScalarRange(min=ScalarLimit(value=0)),
            deterministic=False,
            platform_dependent=True,
            default_value=Observation(
                scalar_double=0,
            ),
        ),
    ]

    def __init__(
        self, working_directory: Path, action_space: ActionSpace, benchmark: Benchmark
    ):
        super().__init__(working_directory, action_space, benchmark)
        logging.info("Started a compilation session for %s", benchmark.uri)
        self._benchmark = benchmark
        self._update_observations()

    def _update_observations(self):
        self._observation = dict()
        # FIXME: "llvm-dis" path is not found. Should we add its path, or is there another way to obtain the IR?
        ir = self._run_command(
            ["llvm-dis", self._benchmark.program.uri, "/dev/stdout"], timeout=5
        )
        self._observation["ir"] = Observation(string_value=ir)

        # TODO: update "features" and "runtime" observations

    # _communicate(...) and _run_command(...) have been copied from llvm_benchmark.py
    # TODO: avoid redundancies and reuse code instead of copying
    def _communicate(self, process, input=None, timeout=None):
        """subprocess.communicate() which kills subprocess on timeout."""
        try:
            return process.communicate(input=input, timeout=timeout)
        except subprocess.TimeoutExpired:
            # kill() was added in Python 3.7.
            if sys.version_info >= (3, 7, 0):
                process.kill()
            else:
                process.terminate()
            raise

    def _run_command(self, cmd: List[str], timeout: int):
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        stdout, stderr = self._communicate(process, timeout=timeout)
        if process.returncode:
            returncode = process.returncode
            try:
                # Try and decode the name of a signal. Signal returncodes
                # are negative.
                returncode = f"{returncode} ({Signals(abs(returncode)).name})"
            except ValueError:
                pass
            raise BenchmarkInitError(
                f"Compilation job failed with returncode {returncode}\n"
                f"Command: {' '.join(cmd)}\n"
                f"Stderr: {stderr.strip()}"
            )
        return stdout

    def apply_action(self, action: Action) -> Tuple[bool, Optional[ActionSpace], bool]:
        logging.info("Applied action %d", action.action)
        if action.action < 0 or action.action > len(self.action_spaces[0].action):
            raise ValueError("Out-of-range")
        return False, None, False

    def get_observation(self, observation_space: ObservationSpace) -> Observation:
        logging.info("Computing observation from space %s", observation_space)
        if observation_space.name == "ir":
            return self._observation["ir"]
        elif observation_space.name == "features":
            observation = Observation()
            observation.int64_list.value[:] = [0, 0, 0]
            return observation
        elif observation_space.name == "runtime":
            return Observation(scalar_double=0)
        else:
            raise KeyError(observation_space.name)


if __name__ == "__main__":
    create_and_run_compiler_gym_service(UnrollingCompilationSession)
