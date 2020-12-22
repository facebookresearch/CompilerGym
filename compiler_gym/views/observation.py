# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Callable, Dict, List

import numpy as np
from gym.spaces import Box, Space

from compiler_gym.service import observation2py, observation_t, scalar_range2tuple
from compiler_gym.service.proto import Observation, ObservationRequest, ObservationSpace
from compiler_gym.spaces import Sequence


class ObservationView(object):
    """A view into the available observation spaces of a service.

    Example usage:

    >>> env = gym.make("llvm-v0")
    >>> env.reset()
    >>> env.observation.spaces.keys()
    ["Autophase", "Ir"]
    >>> env.observation.spaces["Autophase"]
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
        self.indices = {s.name: i for i, s in enumerate(spaces)}
        self.spaces = {s.name: shape2space(s) for s in spaces}
        self.session_id = -1

        self._get_observation = get_observation
        self._base_spaces: Dict[str, Space] = {}
        self._translate_cbs: Dict[str, Callable[[observation_t], observation_t]] = {}

    def __getitem__(self, observation_space: str) -> observation_t:
        """Request an observation from the given space.

        :param observation_space: The observation space to query.
        :return: An observation.
        :raises KeyError: If the requested observation space does not exist.
        """
        request = ObservationRequest(
            session_id=self.session_id,
            observation_space=self.indices[observation_space],
        )
        return self.translate(
            observation_space,
            observation2py(
                self._base_spaces.get(
                    observation_space, self.spaces[observation_space]
                ),
                self._get_observation(request),
            ),
        )

    # TODO(cummins): Register an opaque_data_format handler that replaces the
    # "Space" and updates observation2py / observation2str.

    def register_derived_space(
        self,
        base_name: str,
        derived_name: str,
        derived_space: Space,
        cb: Callable[[observation_t], observation_t],
    ) -> None:
        """Add a hook for implementing derived observation spaces.

        Subclasses of ObservationView call this method in their
        :code:`__init__()` after initializing the base class to register new
        observation spaces that are derived from those provided by the
        CompilerService.

        Example usage:

        Suppose we have a service that provides a "src" observation space
        that returns a string of source code. We want to create a new
        observation space, "src_len", that returns the length of the source
        code. We do this by calling :code:`register_derived_space()` and
        providing the a callback to translate from the base observation space
        to the derived value:

        .. code-block:: python

            class MyObservationView(ObservationView):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.register_derived_space(
                        derived_name="src_len",
                        base_name="src",
                        derived_space=Box(low=0, high=float("inf"), shape=(1,), dtype=int),
                        derive=lambda src: [len(src)],
                    )

        Now we can request for "src_len" observation space and receive
        observations from this new derived space.

        >>> env.observation["src_len"]
        [1021,]

        :param base_name: The name of the observation space that this new
            observation space is derived from.
        :param derived_name: The name of the derived observation space
        """
        self.spaces[derived_name] = derived_space
        self.indices[derived_name] = self.indices[base_name]
        self._translate_cbs[derived_name] = cb

    def __repr__(self):
        return f"ObservationView[{', '.join(sorted(self.spaces.keys()))}]"

    def translate(
        self, observation_space: str, observation: observation_t
    ) -> observation_t:
        """Translate an observation according to the space.

        This methods translates the value returned by a CompilerSpace according
        to any derived observation spaces, as registered using
        register_derived_space(). If the requested observation space is not
        derived the observation is returned unmodified.

        :param observation_space: The name of the observation space.
        :param observation: An observation returned by a CompilerService.
        :return: An observation, after applying any derived space translations.
        """
        return self._translate_cbs.get(observation_space, lambda x: x)(observation)


def shape2space(space: ObservationSpace) -> Space:
    """Convert an ObservationSpace description into a gym Space."""

    def make_box(scalar_range_list, dtype, defaults):
        bounds = [scalar_range2tuple(r, defaults) for r in scalar_range_list]
        return Box(
            low=np.array([b[0] for b in bounds], dtype=dtype),
            high=np.array([b[1] for b in bounds], dtype=dtype),
            dtype=dtype,
        )

    def make_seq(scalar_range, dtype, defaults):
        return Sequence(
            size_range=scalar_range2tuple(scalar_range, defaults),
            dtype=dtype,
            opaque_data_format=space.opaque_data_format,
        )

    shape_type = space.WhichOneof("shape")
    if shape_type == "int64_range_list":
        return make_box(
            space.int64_range_list.range,
            np.int64,
            (np.iinfo(np.int64).min, np.iinfo(np.int64).max),
        )
    elif shape_type == "double_range_list":
        return make_box(space.double_range_list.range, np.float64, (-np.inf, np.inf))
    elif shape_type == "string_size_range":
        return make_seq(space.string_size_range, str, (0, None))
    elif shape_type == "binary_size_range":
        return make_seq(space.binary_size_range, bytes, (0, None))
    else:
        raise TypeError(f"Cannot determine shape of ObservationSpace: {space}")
