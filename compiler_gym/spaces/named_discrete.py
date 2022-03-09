# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections.abc import Iterable as IterableType
from typing import Iterable, List, Union

from compiler_gym.spaces.discrete import Discrete
from compiler_gym.util.gym_type_hints import ActionType


class NamedDiscrete(Discrete):
    """An extension of the :code:`Discrete` space in which each point in the
    space has a name. Additionally, the space itself may have a name.

    :ivar name: The name of the space.
    :vartype name: str

    :ivar names: A list of names for each element in the space.
    :vartype names: List[str]

    Example usage:

    >>> space = NamedDiscrete(["a", "b", "c"])
    >>> space.n
    3
    >>> space["a"]
    0
    >>> space.names[0]
    a
    >>> space.sample()
    1
    """

    def __init__(self, items: Iterable[str], name: str):
        """Constructor.

        :param items: A list of names for items in the space.
        :param name: The name of the space.
        """
        self.names = [str(x) for x in items]
        super().__init__(n=len(self.names), name=name)

    def __getitem__(self, name: str) -> int:
        """Lookup the numeric value of a point in the space.

        :param name: A name.
        :return: The numeric value.
        :raises ValueError: If the name is not in the space.
        """
        return self.names.index(name)

    def __repr__(self) -> str:
        return f"NamedDiscrete([{', '.join(self.names)}])"

    def to_string(self, values: Union[int, Iterable[ActionType]]) -> str:
        """Convert an action, or sequence of actions, to string.

        :param values: A numeric value, or list of numeric values.
        :return: A string representing the values.
        """
        if isinstance(values, IterableType):
            return " ".join([self.names[v] for v in values])
        else:
            return self.names[values]

    def from_string(
        self, values: Union[str, Iterable[str]]
    ) -> Union[ActionType, List[ActionType]]:
        """Convert a name, or list of names, to numeric values.

        :param values: A name, or list of names.
        :return: A numeric value, or list of numeric values.
        """
        if isinstance(values, str):
            return self.names.index(values)
        else:
            return [self.names.index(v) for v in values]
