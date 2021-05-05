# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from compiler_gym.service.proto import (
    ActionSpace,
    Benchmark,
    Observation,
    ObservationSpace,
)


class BaseCompilationSession:
    """Base class for a compilation session."""

    def __init__(
        self, working_directory: Path, action_space_index: int, benchmark: Benchmark
    ):
        del benchmark
        self.working_directory = working_directory
        self.action_space = self.action_spaces[action_space_index]

    def apply_action(self, action: int) -> Tuple[bool, Optional[ActionSpace], bool]:
        raise NotImplementedError

    def apply_actions(
        self, actions: List[int]
    ) -> Tuple[bool, Optional[ActionSpace], bool]:
        end_of_session = False
        new_action_space: Optional[ActionSpace] = None
        actions_had_no_effect = True
        for action in actions:
            end_of_session, nas, ahno = self.apply_action(action)
            new_action_space = nas or new_action_space
            actions_had_no_effect = actions_had_no_effect and ahno
            if end_of_session:
                break

        return end_of_session, new_action_space, actions_had_no_effect


class CompilationSession(BaseCompilationSession):
    """Base class for a compilation session."""

    compiler_version: str = ""  # compiler version
    action_spaces: List[ActionSpace] = []  # what your compiler can do
    observation_spaces: List[ObservationSpace] = []  # what features you provide

    def __init__(
        self, working_directory: Path, action_space_index: int, benchmark: Benchmark
    ):
        # start a new compilation session
        super().__init__(working_directory, action_space_index, benchmark)

    def apply_action(self, action: int) -> Tuple[bool, Optional[ActionSpace], bool]:
        """Return a tuple (end_of_session, new_action_space, action_had_no_effect)."""
        raise NotImplementedError

    def get_observation(self, observation_space_index: int) -> Observation:
        raise NotImplementedError  # compute an observation

    def apply_actions(
        self, actions: Iterable[int]
    ) -> Tuple[bool, Optional[ActionSpace], bool]:
        """Return a tuple (end_of_session, new_action_space, action_had_no_effect)."""
        return super().apply_actions(actions)
