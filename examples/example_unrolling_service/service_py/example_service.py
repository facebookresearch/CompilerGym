#! /usr/bin/env python3
#
#  Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""An example CompilerGym service in python."""
import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

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
        self._action_space = action_space
        self.reset()

    def reset(self):
        self._observation = dict()

        src_uri_p = urlparse(self._benchmark.program.uri)
        self._src_path = os.path.abspath(os.path.join(src_uri_p.netloc, src_uri_p.path))
        # TODO: populate "timestamp" and "benchmark_name" in the path
        # TODO: add "clean_up" function to remove files and save space
        self._benchmark_log_dir = (
            "/tmp/compiler_gym/timestamp/unrolling/benchmark_name/"
        )
        os.makedirs(self._benchmark_log_dir, exist_ok=True)
        self._llvm_path = os.path.join(self._benchmark_log_dir, "version1.ll")
        self._obj_path = os.path.join(self._benchmark_log_dir, "version1.o")
        self._exe_path = os.path.join(self._benchmark_log_dir, "version1")
        # FIXME: llvm.clang_path() lead to build errors
        os.system(
            f"clang -Xclang -disable-O0-optnone -emit-llvm -S {self._src_path} -o {self._llvm_path}"
        )

    def apply_action(self, action: Action) -> Tuple[bool, Optional[ActionSpace], bool]:
        logging.info("Applied action %d", action.action)
        if action.action < 0 or action.action > len(self.action_spaces[0].action):
            raise ValueError("Out-of-range")

        os.system(
            f"{llvm.opt_path()} {self._action_space.action[action.action]} {self._llvm_path} -S -o {self._llvm_path}"
        )
        ir = open(self._llvm_path).read()
        self._observation["ir"] = Observation(string_value=ir)
        return False, None, False  # TODO: return correct values

    def get_observation(self, observation_space: ObservationSpace) -> Observation:
        logging.info("Computing observation from space %s", observation_space)
        if observation_space.name == "ir":
            ir = open(self._llvm_path).read()
            return Observation(string_value=ir)
        elif observation_space.name == "features":
            observation = Observation()
            observation.int64_list.value[:] = [0, 0, 0]
            return observation
        elif observation_space.name == "runtime":
            # TODO: use perf to measure time as it is more accurate
            os.system(
                f"{llvm.llc_path()} -filetype=obj {self._llvm_path} -o {self._obj_path}"
            )
            os.system(f"clang {self._llvm_path} -o {self._exe_path}")
            # FIXME: this is a very inaccurate way to measure time
            start_time = time.time()
            os.system(f"{self._exe_path}")
            exec_time = time.time() - start_time
            return Observation(scalar_double=exec_time)
        else:
            raise KeyError(observation_space.name)


if __name__ == "__main__":
    create_and_run_compiler_gym_service(UnrollingCompilationSession)
