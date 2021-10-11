#! /usr/bin/env python3
#
#  Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""An example CompilerGym service in python."""
import filecmp
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import utils  # TODO: return back its contents to this file?

import compiler_gym.third_party.llvm as llvm
from compiler_gym.service import CompilationSession
from compiler_gym.service.proto import (
    Action,
    ActionSpace,
    Benchmark,
    Observation,
    ObservationSpace,
    ScalarLimit,
    ScalarRange,
    ScalarRangeList,
)
from compiler_gym.service.runtime import create_and_run_compiler_gym_service
from compiler_gym.util.commands import run_command
from compiler_gym.util.timer import Timer


class UnrollingCompilationSession(CompilationSession):
    """Represents an instance of an interactive compilation session."""

    compiler_version: str = "1.0.0"

    # The list of actions that are supported by this service.
    action_spaces = [
        ActionSpace(
            name="unrolling",
            action=[
                "-loop-unroll -unroll-count=2",
                "-loop-unroll -unroll-count=4",
                "-loop-unroll -unroll-count=8",
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
                    ScalarRange(min=ScalarLimit(value=0), max=ScalarLimit(value=1e5)),
                    ScalarRange(min=ScalarLimit(value=0), max=ScalarLimit(value=1e5)),
                    ScalarRange(min=ScalarLimit(value=0), max=ScalarLimit(value=1e5)),
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
        ObservationSpace(
            name="size",
            scalar_double_range=ScalarRange(min=ScalarLimit(value=0)),
            deterministic=True,
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
        self._action_space = action_space

        src_uri_p = urlparse(self._benchmark.program.uri)
        self._src_path = os.path.abspath(os.path.join(src_uri_p.netloc, src_uri_p.path))
        benchmark_name = os.path.basename(self._src_path).split(".")[0]  # noqa
        # TODO: assert that the path exists

        self._llvm_path = os.path.join(self.working_dir, "{benchmark_name}.ll")
        self._obj_path = os.path.join(self.working_dir, "{benchmark_name}.o")
        self._exe_path = os.path.join(self.working_dir, "{benchmark_name}")
        # FIXME: llvm.clang_path() lead to build errors
        run_command(
            [
                "clang",
                "-Xclang",
                "-disable-O0-optnone",
                "-emit-llvm",
                "-S",
                self._src_path,
                "-o",
                self._llvm_path,
            ],
            timeout=30,
        )

    def apply_action(self, action: Action) -> Tuple[bool, Optional[ActionSpace], bool]:
        logging.info("Applied action %d", action.action)
        if action.action < 0 or action.action > len(self.action_spaces[0].action):
            raise ValueError("Out-of-range")

        # make a tmp copy of the LLVM file to compare its contents after applying the action
        llvm_path_before = tempfile.NamedTemporaryFile().name
        shutil.copyfile(self._llvm_path, llvm_path_before)

        os.system(
            f"{llvm.opt_path()} {self._action_space.action[action.action]} {self._llvm_path} -S -o {self._llvm_path}"
        )

        end_of_session = False  # TODO: this needs investigation: for how long can we apply loop unrolling? e.g., detect if there are no more loops in the IR?
        new_action_space = None
        action_had_no_effect = filecmp.cmp(
            llvm_path_before, self._llvm_path, shallow=False
        )
        return (end_of_session, new_action_space, action_had_no_effect)

    def get_observation(self, observation_space: ObservationSpace) -> Observation:
        logging.info("Computing observation from space %s", observation_space)
        if observation_space.name == "ir":
            ir = open(self._llvm_path).read()
            return Observation(string_value=ir)
        elif observation_space.name == "features":
            ir = open(self._llvm_path).read()
            stats = utils.extract_statistics_from_ir(ir)
            observation = Observation()
            observation.int64_list.value[:] = list(stats.values())
            return observation
        elif observation_space.name == "runtime":
            # compile LLVM to object file
            run_command(
                [
                    llvm.llc_path(),
                    "-filetype=obj",
                    self._llvm_path,
                    "-o",
                    self._obj_path,
                ],
                timeout=30,
            )

            # build object file to binary
            run_command(
                [
                    "clang",
                    self._obj_path,
                    "-O3",
                    "-o",
                    self._exe_path,
                ],
                timeout=30,
            )

            # TODO: use more accurate methods to measure time
            # Running 5 times and taking the average of middle 3
            exec_times = []
            for _ in range(5):
                with Timer() as exec_time:
                    run_command(
                        [self._exe_path],
                        timeout=30,
                    )
                exec_times.append(exec_time.time)
            exec_times = np.sort(exec_times)
            avg_exec_time = np.mean(exec_times[1:4])
            return Observation(scalar_double=avg_exec_time)
        elif observation_space.name == "size":
            # compile LLVM to object file
            run_command(
                [
                    llvm.llc_path(),
                    "-filetype=obj",
                    self._llvm_path,
                    "-o",
                    self._obj_path,
                ],
                timeout=30,
            )

            # build object file to binary
            run_command(
                [
                    "clang",
                    self._obj_path,
                    "-Oz",
                    "-o",
                    self._exe_path,
                ],
                timeout=30,
            )
            binary_size = os.path.getsize(self._exe_path)
            return Observation(scalar_double=binary_size)
        else:
            raise KeyError(observation_space.name)


if __name__ == "__main__":
    create_and_run_compiler_gym_service(UnrollingCompilationSession)
