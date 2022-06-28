# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Optional

from gym.spaces import Space

from compiler_gym.util.gym_type_hints import ActionType


class ActionSpace(Space):
    """A wrapper around a :code:`gym.spaces.Space` with additional functionality
    for action spaces.
    """

    def __init__(self, space: Space):
        """Constructor.

        :param space: The space that this action space wraps.
        """
        self.wrapped = space

    def __getattr__(self, name: str):
        return getattr(self.wrapped, name)

    def __getitem__(self, name: str):
        return self.wrapped[name]

    def sample(self) -> ActionType:
        return self.wrapped.sample()

    def seed(self, seed: Optional[int] = None) -> ActionType:
        return self.wrapped.seed(seed)

    def contains(self, x: ActionType) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        raise self.wrapped.contains(x)

    def __contains__(self, x: ActionType) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        return self.wrapped.contains(x)

    def __eq__(self, rhs) -> bool:
        if isinstance(rhs, ActionSpace):
            return self.wrapped == rhs.wrapped
        else:
            return self.wrapped == rhs

    def __ne__(self, rhs) -> bool:
        if isinstance(rhs, ActionSpace):
            return self.wrapped != rhs.wrapped
        else:
            return self.wrapped != rhs

    def to_string(self, actions: List[ActionType]) -> str:
        """Render the provided list of actions to a string.

        This method is used to produce a human-readable string to represent a
        sequence of actions. Subclasses may override the default implementation
        to provide custom rendering.

        This is the complement of :meth:`from_string()
        <compiler_gym.spaces.ActionSpace.from_string>`. The two methods
        are bidirectional:

            >>> actions = env.actions
            >>> s = env.action_space.to_string(actions)
            >>> actions == env.action_space.from_string(s)
            True

        :param actions: A list of actions drawn from this space.

        :return: A string representation that can be decoded using
            :meth:`from_string()
            <compiler_gym.spaces.ActionSpace.from_string>`.
        """
        if hasattr(self.wrapped, "to_string"):
            return self.wrapped.to_string(actions)

        return ",".join(str(x) for x in actions)

    def from_string(self, string: str) -> List[ActionType]:
        """Return a list of actions from the given string.

        This is the complement of :meth:`to_string()
        <compiler_gym.spaces.ActionSpace.to_string>`. The two methods are
        bidirectional:

            >>> actions = env.actions
            >>> s = env.action_space.to_string(actions)
            >>> actions == env.action_space.from_string(s)
            True

        :param string: A string.

        :return: A list of actions.
        """
        if hasattr(self.wrapped, "from_string"):
            return self.wrapped.from_string(string)

        return [self.dtype.type(x) for x in string.split(",")]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.wrapped})"
