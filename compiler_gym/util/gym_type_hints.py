# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Dict, Optional, Tuple, TypeVar

# A JSON dictionary.
JsonDictType = Dict[str, Any]

# Type hints for the values returned by gym.Env.step().
ObservationType = TypeVar("ObservationType")
RewardType = float
DoneType = bool
InfoType = JsonDictType
StepType = Tuple[Optional[ObservationType], Optional[RewardType], DoneType, InfoType]
