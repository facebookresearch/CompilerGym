# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Iterable, List, NamedTuple, Optional, Union

from compiler_gym.spaces.named_discrete import NamedDiscrete


class CommandlineFlag(NamedTuple):
    """A single flag in a Commandline space."""

    name: str
    """The name of the flag, e.g. :code:`LoopUnroll`."""

    flag: str
    """The flag string, e.g. :code:`--unroll`."""

    description: str
    """A human-readable description of the flag."""


class Commandline(NamedDiscrete):
    """A :class:`NamedDiscrete <compiler_gym.spaces.NamedDiscrete>` space where
    each element represents a commandline flag.

    Example usage:

        >>> space = Commandline([
            CommandlineFlag("a", "-a", "A flag"),
            CommandlineFlag("b", "-b", "Another flag"),
        ])
        >>> space.n
        2
        >>> space["a"]
        0
        >>> space.names[0]
        a
        >>> space.flags[0]
        -a
        >>> space.descriptions[0]
        A flag
        >>> space.sample()
        1
        >>> space.commandline([0, 1])
        -a -b

    :ivar flags: A list of flag strings.

    :ivar descriptions: A list of flag descriptions.
    """

    def __init__(self, items: Iterable[CommandlineFlag], name: Optional[str] = None):
        """Constructor.

        :param items: The commandline flags that comprise the space.
        :param name: The name of the space.
        """
        items = list(items)
        self.flags = [f.flag for f in items]
        self.descriptions = [f.description for f in items]
        super().__init__([f.flag for f in items], name)

    def __repr__(self) -> str:
        return f"Commandline([{' '.join(self.flags)}])"

    def commandline(self, values: Union[int, Iterable[int]]) -> str:
        """Produce a commandline invocation from a sequence of values.

        :param values: A numeric value from the space, or sequence of values.
        :return: A string commandline invocation.
        """
        if isinstance(values, int):
            return self.flags[values]
        else:
            return " ".join([self.flags[v] for v in values])

    def from_commandline(self, commandline: str) -> List[int]:
        """Produce a sequence of actions from a commandline.

        :param commandline: A string commandline invocation, as produced by
            :func:`commandline() <compiler_gym.spaces.commandline.Commandline.commandline>`.
        :return: A list of action values.
        :raises LookupError: If any of the flags in the commandline are not
            recognized.
        """
        flags = commandline.split()
        values = []
        for flag in flags:
            try:
                values.append(self.flags.index(flag))
            except IndexError:
                raise LookupError(f"Unknown flag: `{flag}`")
        return values
