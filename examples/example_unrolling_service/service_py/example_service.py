#! /usr/bin/env python3
#
#  Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""An example CompilerGym service in python."""
import logging
import os
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

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
        p = urlparse(self._benchmark.program.uri)
        final_path = os.path.abspath(os.path.join(p.netloc, p.path))
        ir = open(final_path).read()
        print("ir: ", ir)
        self._observation["ir"] = Observation(string_value=ir)
        # TODO: update "features" and "runtime" observations

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
