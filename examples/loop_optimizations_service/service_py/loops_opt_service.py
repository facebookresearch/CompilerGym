#! /usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""An example CompilerGym service in python."""
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import utils

import compiler_gym.third_party.llvm as llvm
from compiler_gym.service import CompilationSession
from compiler_gym.service.proto import (
    Action,
    ActionSpace,
    Benchmark,
    ChoiceSpace,
    NamedDiscreteSpace,
    Observation,
    ObservationSpace,
    ScalarLimit,
    ScalarRange,
    ScalarRangeList,
)
from compiler_gym.service.runtime import create_and_run_compiler_gym_service
from compiler_gym.util.commands import run_command


class LoopsOptCompilationSession(CompilationSession):
    """Represents an instance of an interactive compilation session."""

    compiler_version: str = "1.0.0"

    # The list of actions that are supported by this service.
    action_spaces = [
        ActionSpace(
            name="loop-opt",
            choice=[
                ChoiceSpace(
                    name="loop-opt",
                    named_discrete_space=NamedDiscreteSpace(
                        value=[
                            "--loop-unroll --unroll-count=2",
                            "--loop-unroll --unroll-count=4",
                            "--loop-unroll --unroll-count=8",
                            "--loop-unroll --unroll-count=16",
                            "--loop-unroll --unroll-count=32",
                            "--loop-vectorize -force-vector-width=2",
                            "--loop-vectorize -force-vector-width=4",
                            "--loop-vectorize -force-vector-width=8",
                            "--loop-vectorize -force-vector-width=16",
                            "--loop-vectorize -force-vector-width=32",
                        ]
                    ),
                ),
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
        self,
        working_directory: Path,
        action_space: ActionSpace,
        benchmark: Benchmark,
        use_custom_opt: bool = True,
    ):
        super().__init__(working_directory, action_space, benchmark)
        logging.info("Started a compilation session for %s", benchmark.uri)
        self._benchmark = benchmark
        self._action_space = action_space

        # Resolve the paths to LLVM binaries once now.
        self._clang = str(llvm.clang_path())
        self._llc = str(llvm.llc_path())
        self._llvm_diff = str(llvm.llvm_diff_path())
        self._opt = str(llvm.opt_path())
        # LLVM's opt does not always enforce the loop optimization options passed as cli arguments.
        # Hence, we created our own exeutable with custom unrolling and vectorization pass in examples/loops_opt_service/opt_loops that enforces the unrolling and vectorization factors passed in its cli.
        # if self._use_custom_opt is true, use our custom exeutable, otherwise use LLVM's opt
        self._use_custom_opt = use_custom_opt

        # Dump the benchmark source to disk.
        self._src_path = str(self.working_dir / "benchmark.c")
        with open(self.working_dir / "benchmark.c", "wb") as f:
            f.write(benchmark.program.contents)

        self._llvm_path = str(self.working_dir / "benchmark.ll")
        self._llvm_before_path = str(self.working_dir / "benchmark.previous.ll")
        self._obj_path = str(self.working_dir / "benchmark.o")
        self._exe_path = str(self.working_dir / "benchmark.exe")

        run_command(
            [
                self._clang,
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
        num_choices = len(self._action_space.choice[0].named_discrete_space.value)

        if len(action.choice) != 1:
            raise ValueError("Currently we support one choice at a time")

        # This is the index into the action space's values ("a", "b", "c") that
        # the user selected, e.g. 0 -> "a", 1 -> "b", 2 -> "c".
        choice_index = action.choice[0].named_discrete_value_index
        if choice_index < 0 or choice_index >= num_choices:
            raise ValueError("Out-of-range")

        args = self._action_space.choice[0].named_discrete_space.value[choice_index]
        logging.info(
            "Applying action %d, equivalent command-line arguments: '%s'",
            choice_index,
            args,
        )
        args = args.split()

        # make a copy of the LLVM file to compare its contents after applying the action
        shutil.copyfile(self._llvm_path, self._llvm_before_path)

        # apply action
        if self._use_custom_opt:
            # our custom opt-loops has an additional `f` at the beginning of each argument
            for i, arg in enumerate(args):
                # convert --<argument> to --f<argument>
                arg = arg[0:2] + "f" + arg[2:]
                args[i] = arg
            run_command(
                [
                    "../opt_loops/opt_loops",
                    self._llvm_path,
                    *args,
                    "-S",
                    "-o",
                    self._llvm_path,
                ],
                timeout=30,
            )
        else:
            run_command(
                [
                    self._opt,
                    *args,
                    self._llvm_path,
                    "-S",
                    "-o",
                    self._llvm_path,
                ],
                timeout=30,
            )

        # compare the IR files to check if the action had an effect
        try:
            subprocess.check_call(
                [self._llvm_diff, self._llvm_before_path, self._llvm_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60,
            )
            action_had_no_effect = True
        except subprocess.CalledProcessError:
            action_had_no_effect = False

        end_of_session = False  # TODO: this needs investigation: for how long can we apply loop optimizations? e.g., detect if there are no more loops in the IR? or look at the metadata?
        new_action_space = None
        return (end_of_session, new_action_space, action_had_no_effect)

    @property
    def ir(self) -> str:
        with open(self._llvm_path) as f:
            return f.read()

    def get_observation(self, observation_space: ObservationSpace) -> Observation:
        logging.info("Computing observation from space %s", observation_space.name)
        if observation_space.name == "ir":
            return Observation(string_value=self.ir)
        elif observation_space.name == "features":
            stats = utils.extract_statistics_from_ir(self.ir)
            observation = Observation()
            observation.int64_list.value[:] = list(stats.values())
            return observation
        elif observation_space.name == "runtime":
            # compile LLVM to object file
            run_command(
                [
                    self._llc,
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

            # TODO: add documentation that benchmarks need print out execution time
            # Running 5 times and taking the average of middle 3
            exec_times = []
            for _ in range(5):
                stdout = run_command(
                    [self._exe_path],
                    timeout=30,
                )
                try:
                    exec_times.append(int(stdout))
                except ValueError:
                    raise ValueError(
                        f"Error in parsing execution time from output of command\n"
                        f"Please ensure that the source code of the benchmark measures execution time and prints to stdout\n"
                        f"Stdout of the program: {stdout}"
                    )
            exec_times = np.sort(exec_times)
            avg_exec_time = np.mean(exec_times[1:4])
            return Observation(scalar_double=avg_exec_time)
        elif observation_space.name == "size":
            # compile LLVM to object file
            run_command(
                [
                    self._llc,
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
    create_and_run_compiler_gym_service(LoopsOptCompilationSession)
