# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
from typing import Callable, Optional, Union

import networkx as nx
import numpy as np
from gym.spaces import Space

from compiler_gym.service.proto import Observation, ObservationSpace, ScalarRange
from compiler_gym.spaces.box import Box
from compiler_gym.spaces.scalar import Scalar
from compiler_gym.spaces.sequence import Sequence
from compiler_gym.util.gym_type_hints import ObservationType


def _json2nx(observation):
    json_data = json.loads(observation.string_value)
    return nx.readwrite.json_graph.node_link_graph(
        json_data, multigraph=True, directed=True
    )


def _scalar_range2tuple(sr: ScalarRange, defaults=(-np.inf, np.inf)):
    """Convert a ScalarRange to a tuple of (min, max) bounds."""
    return (
        sr.min.value if sr.HasField("min") else defaults[0],
        sr.max.value if sr.HasField("max") else defaults[1],
    )


class ObservationSpaceSpec:
    """Specification of an observation space.

    :ivar id: The name of the observation space.
    :vartype id: str

    :ivar index: The index into the list of observation spaces that the service
        supports.
    :vartype index: int

    :ivar space: The space.
    :vartype space: Space

    :ivar deterministic: Whether the observation space is deterministic.
    :vartype deterministic: bool

    :ivar platform_dependent: Whether the observation values depend on the
        execution environment of the service.
    :vartype platform_dependent: bool

    :ivar default_value: A default observation. This value will be returned by
        :func:`CompilerEnv.step() <compiler_gym.envs.CompilerEnv.step>` if
        :func:`CompilerEnv.observation_space <compiler_gym.envs.CompilerEnv.observation_space>`
        is set and the service terminates.
    """

    def __init__(
        self,
        id: str,
        index: int,
        space: Space,
        translate: Callable[[Union[ObservationType, Observation]], ObservationType],
        to_string: Callable[[ObservationType], str],
        deterministic: bool,
        platform_dependent: bool,
        default_value: ObservationType,
    ):
        """Constructor. Don't call directly, use make_derived_space()."""
        self.id: str = id
        self.index: int = index
        self.space = space
        self.deterministic = deterministic
        self.platform_dependent = platform_dependent
        self.default_value = default_value
        self.translate = translate
        self.to_string = to_string

    def __hash__(self) -> int:
        # Quickly hash observation spaces by comparing the index into the list
        # of spaces returned by the environment. This means that you should not
        # hash between observation spaces from different environments as this
        # will cause collisions, e.g.
        #
        #     # not okay:
        #     >>> obs = set(env.observation.spaces).union(
        #         other_env.observation.spaces
        #     )
        #
        # If you want to hash between environments, consider using the string id
        # to identify the observation spaces.
        return self.index

    def __repr__(self) -> str:
        return f"ObservationSpaceSpec({self.id})"

    def __eq__(self, rhs) -> bool:
        """Equality check."""
        if not isinstance(rhs, ObservationSpaceSpec):
            return False
        return (
            self.id == rhs.id
            and self.index == rhs.index
            and self.space == rhs.space
            and self.platform_dependent == rhs.platform_dependent
            and self.deterministic == rhs.deterministic
        )

    @classmethod
    def from_proto(cls, index: int, proto: ObservationSpace):
        """Construct a space from an ObservationSpace message."""
        shape_type = proto.WhichOneof("shape")

        def make_box(scalar_range_list, dtype, defaults):
            bounds = [_scalar_range2tuple(r, defaults) for r in scalar_range_list]
            return Box(
                name=proto.name,
                low=np.array([b[0] for b in bounds], dtype=dtype),
                high=np.array([b[1] for b in bounds], dtype=dtype),
                dtype=dtype,
            )

        def make_scalar(scalar_range, dtype, defaults):
            scalar_range_tuple = _scalar_range2tuple(scalar_range, defaults)
            return Scalar(
                name=proto.name,
                min=dtype(scalar_range_tuple[0]),
                max=dtype(scalar_range_tuple[1]),
                dtype=dtype,
            )

        def make_seq(size_range, dtype, defaults, scalar_range=None):
            return Sequence(
                name=proto.name,
                size_range=_scalar_range2tuple(size_range, defaults),
                dtype=dtype,
                opaque_data_format=proto.opaque_data_format,
                scalar_range=scalar_range,
            )

        # Translate from protocol buffer specification to python. There are
        # three variables to derive:
        #   (1) space: the gym.Space instance describing the space.
        #   (2) translate: is a callback that translates from an Observation
        #           message to a python type.
        #   (3) to_string: is a callback that translates from a python type to a
        #           string for printing.
        if proto.opaque_data_format == "json://networkx/MultiDiGraph":
            # TODO(cummins): Add a Graph space.
            space = make_seq(proto.string_size_range, str, (0, None))

            def translate(observation):
                return nx.readwrite.json_graph.node_link_graph(
                    json.loads(observation.string_value), multigraph=True, directed=True
                )

            def to_string(observation):
                return json.dumps(
                    nx.readwrite.json_graph.node_link_data(observation), indent=2
                )

        elif proto.opaque_data_format == "json://":
            space = make_seq(proto.string_size_range, str, (0, None))

            def translate(observation):
                return json.loads(observation.string_value)

            def to_string(observation):
                return json.dumps(observation, indent=2)

        elif shape_type == "int64_range_list":
            space = make_box(
                proto.int64_range_list.range,
                np.int64,
                (np.iinfo(np.int64).min, np.iinfo(np.int64).max),
            )

            def translate(observation):
                return np.array(observation.int64_list.value, dtype=np.int64)

            to_string = str
        elif shape_type == "double_range_list":
            space = make_box(
                proto.double_range_list.range, np.float64, (-np.inf, np.inf)
            )

            def translate(observation):
                return np.array(observation.double_list.value, dtype=np.float64)

            to_string = str
        elif shape_type == "string_size_range":
            space = make_seq(proto.string_size_range, str, (0, None))

            def translate(observation):
                return observation.string_value

            to_string = str
        elif shape_type == "binary_size_range":
            space = make_seq(proto.binary_size_range, bytes, (0, None))

            def translate(observation):
                return observation.binary_value

            to_string = str
        elif shape_type == "scalar_int64_range":
            space = make_scalar(
                proto.scalar_int64_range,
                int,
                (np.iinfo(np.int64).min, np.iinfo(np.int64).max),
            )

            def translate(observation):
                return int(observation.scalar_int64)

            to_string = str
        elif shape_type == "scalar_double_range":
            space = make_scalar(proto.scalar_double_range, float, (-np.inf, np.inf))

            def translate(observation):
                return float(observation.scalar_double)

            to_string = str
        elif shape_type == "double_sequence":
            space = make_seq(
                proto.double_sequence.length_range,
                np.float64,
                (-np.inf, np.inf),
                make_scalar(
                    proto.double_sequence.scalar_range, np.float64, (-np.inf, np.inf)
                ),
            )

            def translate(observation):
                return np.array(observation.double_list.value, dtype=np.float64)

            to_string = str
        else:
            raise TypeError(
                f"Unknown shape '{shape_type}' for ObservationSpace:\n{proto}"
            )

        return cls(
            id=proto.name,
            index=index,
            space=space,
            translate=translate,
            to_string=to_string,
            deterministic=proto.deterministic,
            platform_dependent=proto.platform_dependent,
            default_value=translate(proto.default_value),
        )

    def make_derived_space(
        self,
        id: str,
        translate: Callable[[ObservationType], ObservationType],
        space: Optional[Space] = None,
        deterministic: Optional[bool] = None,
        default_value: Optional[ObservationType] = None,
        platform_dependent: Optional[bool] = None,
        to_string: Callable[[ObservationType], str] = None,
    ) -> "ObservationSpaceSpec":
        """Create a derived observation space.

        :param id: The name of the derived observation space.
        :param translate: A callback function to compute a derived observation
            from the base observation.
        :param space: The :code:`gym.Space` describing the observation space.
        :param deterministic: Whether the observation space is deterministic.
            If not provided, the value is inherited from the base observation
            space.
        :param default_value: The default value for the observation space. If
            not provided, the value is derived from the default value of the
            base observation space.
        :param platform_dependent: Whether the derived observation space is
            platform-dependent. If not provided, the value is inherited from
            the base observation space.
        :param to_string: A callback to convert and observation to a string
            representation. If not provided, the callback is inherited from the
            base observation space.
        :return: A new ObservationSpaceSpec.
        """
        return ObservationSpaceSpec(
            id=id,
            index=self.index,
            space=space or self.space,
            translate=lambda observation: translate(self.translate(observation)),
            to_string=to_string or self.to_string,
            default_value=(
                translate(self.default_value)
                if default_value is None
                else default_value
            ),
            deterministic=(
                self.deterministic if deterministic is None else deterministic
            ),
            platform_dependent=(
                self.platform_dependent
                if platform_dependent is None
                else platform_dependent
            ),
        )
