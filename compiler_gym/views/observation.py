# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Callable, Dict, List

from compiler_gym.service import ServiceError
from compiler_gym.service.proto import ObservationSpace, StepReply, StepRequest
from compiler_gym.util.gym_type_hints import ObservationType
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
        get_observation: Callable[[StepRequest], StepReply],
        spaces: List[ObservationSpace],
    ):
        if not spaces:
            raise ValueError("No observation spaces")
        self.spaces: Dict[str, ObservationSpaceSpec] = {}

        self._get_observation = get_observation
        self.session_id = -1

        for i, s in enumerate(spaces):
            self._add_space(ObservationSpaceSpec.from_proto(i, s))

    def __getitem__(self, observation_space: str) -> ObservationType:
        """Request an observation from the given space.

        :param observation_space: The observation space to query.

        :return: An observation.

        :raises KeyError: If the requested observation space does not exist.

        :raises SessionNotFound: If :meth:`env.reset()
            <compiler_gym.envs.CompilerEnv.reset>` has not been called.
        """
        space = self.spaces[observation_space]
        request = StepRequest(
            session_id=self.session_id,
            observation_space=[space.index],
        )
        reply: StepReply = self._get_observation(request)
        if len(reply.observation) != 1:
            raise ServiceError(
                f"Requested 1 observation but received {len(reply.observation)}"
            )
        return space.translate(reply.observation[0])

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
        """Alias to :func:`ObservationSpaceSpec.make_derived_space()
        <compiler_gym.views.ObservationSpaceSpec.make_derived_space>` that adds
        the derived space to the observation view.

        Example usage:

            >>> env.observation.add_derived_space(
                id="src_len",
                base_id="src",
                translate=lambda src: np.array([len(src)], dtype=np.int32),
                shape=Box(shape=(1,), dtype=np.int32),
            )
            >>> env.observation["src_len"]
            1029

        :param id: The name of the new observation space.

        :param base_id: The name of the observation space that this is derived
            from.

        :param \\**kwargs: Arguments passed to
            :func:`ObservationSpaceSpec.make_derived_space
            <compiler_gym.views.ObservationSpaceSpec.make_derived_space>`.
        """
        base_space = self.spaces[base_id]
        self._add_space(base_space.make_derived_space(id=id, **kwargs))

    def __repr__(self):
        return f"ObservationView[{', '.join(sorted(self.spaces.keys()))}]"
