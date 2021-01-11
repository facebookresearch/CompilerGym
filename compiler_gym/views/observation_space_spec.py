# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
from typing import Callable, Optional, Union

import networkx as nx
import numpy as np
from gym.spaces import Box, Space

from compiler_gym.service import observation_t, scalar_range2tuple
from compiler_gym.service.proto import Observation, ObservationSpace
from compiler_gym.spaces import Sequence


def _json2nx(observation):
    json_data = json.loads(observation.string_value)
    return nx.readwrite.json_graph.node_link_graph(
        json_data, multigraph=True, directed=True
    )


class ObservationSpaceSpec(object):
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
        cb: Callable[[Union[observation_t, Observation]], observation_t],
        to_string: Callable[[observation_t], str],
        deterministic: bool,
        platform_dependent: bool,
        default_value: observation_t,
    ):
        """Constructor. Don't call directly, use make_derived_space()."""
        self.id: str = id
        self.index: int = index
        self.space = space
        self.deterministic = deterministic
        self.platform_dependent = platform_dependent
        self.default_value = default_value
        self.cb = cb
        self.to_string = to_string

    def __repr__(self) -> str:
        return f"ObservationSpaceSpec({self.id})"

    @classmethod
    def from_proto(cls, index: int, proto: ObservationSpace):
        """Construct a space from an ObservationSpace message."""
        shape_type = proto.WhichOneof("shape")

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
                opaque_data_format=proto.opaque_data_format,
            )

        # Translate from protocol buffer specification to python. There are
        # three variables to derive: 'space', the gym.Space instance describing
        # the space. 'cb' is a callback that translates from an Observation
        # message to a python type. and 'to_string' is a callback that
        # translates from a python type to a string for printing.
        if proto.opaque_data_format == "json://networkx/MultiDiGraph":
            # TODO(cummins): Add a Graph space.
            space = make_seq(proto.string_size_range, str, (0, None))
            cb = lambda observation: nx.readwrite.json_graph.node_link_graph(
                json.loads(observation.string_value), multigraph=True, directed=True
            )
            to_string = lambda observation: json.dumps(
                nx.readwrite.json_graph.node_link_data(observation), indent=2
            )
        elif proto.opaque_data_format == "json://":
            space = make_seq(proto.string_size_range, str, (0, None))
            cb = lambda observation: json.loads(observation.string_value)
            to_string = lambda observation: json.dumps(observation, indent=2)
        elif shape_type == "int64_range_list":
            space = make_box(
                proto.int64_range_list.range,
                np.int64,
                (np.iinfo(np.int64).min, np.iinfo(np.int64).max),
            )
            cb = lambda observation: np.array(
                observation.int64_list.value, dtype=np.int64
            )
            to_string = str
        elif shape_type == "double_range_list":
            space = make_box(
                proto.double_range_list.range, np.float64, (-np.inf, np.inf)
            )
            cb = lambda observation: np.array(
                observation.double_list.value, dtype=np.float64
            )
            to_string = str
        elif shape_type == "string_size_range":
            space = make_seq(proto.string_size_range, str, (0, None))
            cb = lambda observation: observation.string_value
            to_string = str
        elif shape_type == "binary_size_range":
            space = make_seq(proto.binary_size_range, bytes, (0, None))
            cb = lambda observation: observation.binary_value
            to_string = str
        else:
            raise TypeError(f"Cannot determine shape of ObservationSpace: {proto}")

        return cls(
            id=proto.name,
            index=index,
            space=space,
            cb=cb,
            to_string=to_string,
            deterministic=proto.deterministic,
            platform_dependent=proto.platform_dependent,
            default_value=cb(proto.default_value),
        )

    def make_derived_space(
        self,
        id: str,
        cb: Callable[[observation_t], observation_t],
        space: Optional[Space] = None,
        deterministic: Optional[bool] = None,
        default_value: Optional[observation_t] = None,
        platform_dependent: Optional[bool] = None,
        to_string: Callable[[observation_t], str] = None,
    ) -> "ObservationSpaceSpec":
        """Create a derived observation space.

        :param id: The name of the derived observation space.
        :param cb: A callback function to compute a derived observation
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
            cb=lambda observation: cb(self.cb(observation)),
            to_string=to_string or self.to_string,
            default_value=(
                cb(self.default_value) if default_value is None else default_value
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
