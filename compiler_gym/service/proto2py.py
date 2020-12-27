# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Converters from protocol buffers to python-friendly types."""
import json
from typing import Any, Dict, List, Union

import networkx as nx
import numpy as np
from gym.spaces import Space

from compiler_gym.service.proto import Observation, RewardSpace, ScalarRange
from compiler_gym.spaces import Sequence

json_t = Union[List[Any], Dict[str, Any]]
observation_t = Union[np.ndarray, str, bytes, json_t, nx.DiGraph]


def scalar_range2tuple(sr: ScalarRange, defaults=(-np.inf, np.inf)):
    """Convert a ScalarRange to a tuple of (min, max) bounds."""
    return (
        sr.min.value if sr.HasField("min") else defaults[0],
        sr.max.value if sr.HasField("max") else defaults[1],
    )


def _opaque2py(opaque_data_format: str, observation: Observation) -> nx.DiGraph:
    """Deserialize an "opaque" message (i.e. a message that requires interpretation)."""
    if opaque_data_format == "json://networkx/MultiDiGraph":
        json_data = json.loads(observation.string_value)
        return nx.readwrite.json_graph.node_link_graph(
            json_data, multigraph=True, directed=True
        )
    elif opaque_data_format == "json://":
        json_data = json.loads(observation.string_value)
        return json_data
    else:
        raise TypeError(f"Unknown data format for observation: {opaque_data_format}")


def observation2py(
    space: Space,
    observation: Observation,
) -> observation_t:
    """Convert an Observation message to a python-friendly type.

    Numeric lists are converted to numpy arrays, opaque sequences are returned
    as strings / bytes.
    """
    # TODO(cummins): merge into ObservationView class.
    if not isinstance(observation, Observation):
        raise TypeError(
            f"Did not receive an Observation object, received {observation.__class__}"
        )
    if isinstance(space, Sequence) and space.opaque_data_format:
        return _opaque2py(space.opaque_data_format, observation)
    if observation.HasField("int64_list"):
        return np.array(observation.int64_list.value, dtype=np.int64)
    elif observation.HasField("double_list"):
        return np.array(observation.double_list.value, dtype=np.float64)
    elif observation.HasField("string_value"):
        return observation.string_value
    elif observation.HasField("binary_value"):
        return observation.binary_value
    else:
        raise TypeError(f"Unknown Observation type: {observation}")


def observation2str(space: Space, observation: observation_t) -> str:
    if (
        isinstance(space, Sequence)
        and space.opaque_data_format == "json://networkx/MultiDiGraph"
    ):
        return json.dumps(nx.readwrite.json_graph.node_link_data(observation), indent=2)
    else:
        return str(observation)
