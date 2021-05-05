from typing import List, Optional, Tuple

from compiler_gym.service.proto import (
    ActionSpace,
    Benchmark,
    Observation,
    ObservationSpace,
)


class BaseCompilationSession:
    """Base class for a compilation session."""

    def apply_actions(
        self, actions: List[int]
    ) -> Tuple[bool, Optional[ActionSpace], bool]:
        # optional.
        end_of_session = False
        new_action_space = None
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

    action_spaces: List[ActionSpace] = []  # what your compiler can do
    observation_spaces: List[ObservationSpace] = []  # what features you provide

    def __init__(self, action_space_index: int, benchmark: Benchmark):
        pass  # start a new compilation session

    def apply_action(self, action: int) -> Tuple[bool, Optional[ActionSpace], bool]:
        """Return a tuple (end_of_session, new_action_space, action_had_no_effect)."""
        raise NotImplementedError  # apply an action

    def get_observation(self, observation_space_index: int) -> Observation:
        raise NotImplementedError  # compute an observation
