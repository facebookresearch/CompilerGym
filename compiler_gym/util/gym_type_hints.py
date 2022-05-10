# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

# A JSON dictionary.
JsonDictType = Dict[str, Any]

# A default value for the reward_space parameter in env.reset() and rev.observation_space() functions.
class OptionalArgumentValue(Enum):
    UNCHANGED = 1


# Type hints for the values returned by gym.Env.step().
ObservationType = TypeVar("ObservationType")
ActionType = TypeVar("ActionType")
RewardType = float
DoneType = bool
InfoType = JsonDictType
StepType = Tuple[
    Optional[Union[ObservationType, List[ObservationType]]],
    Optional[Union[RewardType, List[RewardType]]],
    DoneType,
    InfoType,
]
