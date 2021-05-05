#! /usr/bin/env python3
#
#  Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""An example CompilerGym service in python."""
import logging
from pathlib import Path
from typing import Optional, Tuple

from compiler_gym.service.core import CompilationSession
from compiler_gym.service.core.run import create_and_run_compiler_gym_service
from compiler_gym.service.proto import (
    ActionSpace,
    Benchmark,
    Observation,
    ObservationSpace,
    ScalarLimit,
    ScalarRange,
    ScalarRangeList,
)


class ExampleCompilationSession(CompilationSession):
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
        self, working_directory: Path, action_space_index: int, benchmark: Benchmark
    ):
        super().__init__(working_directory, action_space_index, benchmark)
        logging.info("Started a compilation session for %s", benchmark.uri)

    def apply_action(self, action: int) -> Tuple[bool, Optional[ActionSpace], bool]:
        logging.info("Applied action %d", action)
        if action < 0 or action > len(self.action_space.action):
            raise ValueError("Out-of-range")
        return False, None, False

    def get_observation(self, observation_space_index: int) -> Observation:
        observation_space = self.observation_spaces[observation_space_index]
        logging.info("Computing observation from space %d", observation_space_index)
        if observation_space.name == "ir":
            return Observation(string_value="Hello, world!")
        elif observation_space.name == "features":
            observation = Observation()
            observation.int64_list.value[:] = [0, 0, 0]
            return observation
        elif observation_space.name == "runtime":
            return Observation(scalar_double=0)
        else:
            raise KeyError(observation_space_index)


if __name__ == "__main__":
    create_and_run_compiler_gym_service(ExampleCompilationSession)
