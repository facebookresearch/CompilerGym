# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Callable, Dict, List

from compiler_gym.errors import ServiceError
from compiler_gym.service.proto import ObservationSpace
from compiler_gym.util.gym_type_hints import (
    ActionType,
    ObservationType,
    RewardType,
    StepType,
)
from compiler_gym.views.observation_space_spec import ObservationSpaceSpec


class ObservationView:
    """A view into the available observation spaces of a service.

    Example usage:

        >>> env = gym.make("llvm-v0")
        >>> env.reset()
        >>> env.observation.spaces.keys()
        ["Autophase", "Ir"]
        >>> env.observation.spaces["Autophase"].space
        Box(56,)
        >>> env.observation["Autophase"]
        [0, 1, ..., 2]
        >>> observation["Ir"]
        int main() {...}
    """

    def __init__(
        self,
        raw_step: Callable[
            [List[ActionType], List[ObservationType], List[RewardType]], StepType
        ],
        spaces: List[ObservationSpace],
    ):
        if not spaces:
            raise ValueError("No observation spaces")
        self.spaces: Dict[str, ObservationSpaceSpec] = {}

        self._raw_step = raw_step

        for i, s in enumerate(spaces):
            self._add_space(ObservationSpaceSpec.from_proto(i, s))

    def __getitem__(self, observation_space: str) -> ObservationType:
        """Request an observation from the given space.

        :param observation_space: The observation space to query.

        :return: An observation.

        :raises KeyError: If the requested observation space does not exist.

        :raises SessionNotFound: If :meth:`env.reset()
            <compiler_gym.envs.CompilerEnv.reset>` has not been called.

        :raises ServiceError: If the backend service fails to compute the
            observation, or reports that a terminal state has been reached.
        """
        observation_space: ObservationSpaceSpec = self.spaces[observation_space]
        observations, _, done, info = self._raw_step(
            actions=[], observation_spaces=[observation_space], reward_spaces=[]
        )

        if done:
            # Computing an observation should never cause a terminal state since
            # no action has been applied.
            msg = f"Failed to compute observation '{observation_space.id}'"
            if info.get("error_details"):
                msg += f": {info['error_details']}"
            raise ServiceError(msg)

        if len(observations) != 1:
            raise ServiceError(
                f"Expected 1 '{observation_space.id}' observation "
                f"but the service returned {len(observations)}"
            )

        return observations[0]

    def _add_space(self, space: ObservationSpaceSpec):
        """Register a new space."""
        self.spaces[space.id] = space
        # Bind a new method to this class that is a callback to compute the
        # given observation space. E.g. if a new space is added with ID
        # `FooBar`, this observation can be computed using
        # env.observation.FooBar().
        setattr(self, space.id, lambda: self[space.id])

    def add_derived_space(
        self,
        id: str,
        base_id: str,
        **kwargs,
    ) -> None:
        """Internal API for adding a new observation space."""
        base_space = self.spaces[base_id]
        self._add_space(base_space.make_derived_space(id=id, **kwargs))

    def __repr__(self):
        return f"ObservationView[{', '.join(sorted(self.spaces.keys()))}]"
