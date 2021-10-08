# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict as DictType
from typing import List, Union

from gym.spaces import Dict as GymDict
from gym.spaces import Space


class Dict(GymDict):
    """A dictionary of simpler spaces.

    Wraps the underlying :code:`gym.spaces.Dict` space with a name attribute.
    """

    def __init__(self, spaces: Union[DictType[str, Space], List[Space]], name: str):
        """Constructor.

        :param spaces: The composite spaces.

        :param name: The name of the space.
        """
        super().__init__(spaces)
        self.name = name
