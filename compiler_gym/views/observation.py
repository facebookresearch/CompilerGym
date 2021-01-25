# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Callable, List

from compiler_gym.service import observation_t
from compiler_gym.service.proto import Observation, ObservationRequest, ObservationSpace
from compiler_gym.views.observation_space_spec import ObservationSpaceSpec


class ObservationView(object):
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
        get_observation: Callable[[ObservationRequest], Observation],
        spaces: List[ObservationSpace],
    ):
        if not spaces:
            raise ValueError("No observation spaces")
        self.spaces = {
            s.name: ObservationSpaceSpec.from_proto(i, s) for i, s in enumerate(spaces)
        }
        self.session_id = -1

        self._get_observation = get_observation

    def __getitem__(self, observation_space: str) -> observation_t:
        """Request an observation from the given space.

        :param observation_space: The observation space to query.
        :return: An observation.
        :raises KeyError: If the requested observation space does not exist.
        """
        space = self.spaces[observation_space]
        request = ObservationRequest(
            session_id=self.session_id,
            observation_space=space.index,
        )
        return space.cb(self._get_observation(request))

    def add_derived_space(
        self,
        id: str,
        base_id: str,
        **kwargs,
    ) -> None:
        """Alias to
        :func:`ObservationSpaceSpec.make_derived_space() <compiler_gym.views.ObservationSpaceSpec.make_derived_space>`
        that adds the derived space to the observation view.

        Example usage:

        >>> env.observation.add_derived_space(
            id="src_len",
            base_id="src",
            cb=lambda src: np.array([len(src)], dtype=np.int32),
            shape=Box(shape=(1,), dtype=np.int32),
        )
        >>> env.observation["src_len"]
        1029

        :param id: The name of the new observation space.
        :param base_id: The name of the observation space that this is derived
            from.
        :param **kwargs: Arguments passed to
            :func:`ObservationSpaceSpec.make_derived_space <compiler_gym.views.ObservationSpaceSpec.make_derived_space>`.
        """
        base_space = self.spaces[base_id]
        self.spaces[id] = base_space.make_derived_space(id=id, **kwargs)

    def __repr__(self):
        return f"ObservationView[{', '.join(sorted(self.spaces.keys()))}]"
