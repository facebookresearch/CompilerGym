# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Converters from protocol buffers to python-friendly types."""
from typing import Any, Dict, List, Union

import networkx as nx
import numpy as np

from compiler_gym.service.proto import ScalarRange

json_t = Union[List[Any], Dict[str, Any]]
observation_t = Union[np.ndarray, str, bytes, int, float, json_t, nx.DiGraph]


def scalar_range2tuple(sr: ScalarRange, defaults=(-np.inf, np.inf)):
    """Convert a ScalarRange to a tuple of (min, max) bounds."""
    return (
        sr.min.value if sr.HasField("min") else defaults[0],
        sr.max.value if sr.HasField("max") else defaults[1],
    )
