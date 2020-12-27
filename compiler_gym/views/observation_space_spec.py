# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import numpy as np
from gym.spaces import Box, Space

from compiler_gym.service import scalar_range2tuple
from compiler_gym.service.proto import ObservationSpace
from compiler_gym.spaces import Sequence


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
    """

    def __init__(
        self,
        id: str,
        space: Space,
        deterministic: bool = True,
        platform_dependent: bool = False,
    ):
        self.id = id
        self.index: Optional[int] = None
        self.space = space
        self.deterministic = deterministic
        self.platform_dependent = platform_dependent

    def __repr__(self) -> str:
        return f"ObservationSpaceSpec({self.id})"

    @classmethod
    def from_proto(cls, index: int, spec: ObservationSpace):
        ret = cls(
            id=spec.name,
            space=shape2space(spec),
            deterministic=spec.deterministic,
            platform_dependent=spec.platform_dependent,
        )
        ret.index = index
        return ret


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
