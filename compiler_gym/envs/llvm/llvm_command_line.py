# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

from compiler_gym.spaces import ActionSpace
from compiler_gym.util.gym_type_hints import ActionType


class LlvmCommandLine(ActionSpace):
    """An action space for LLVM that supports serializing / deserializing to
    opt command line.
    """

    def to_string(self, actions: List[ActionType]) -> str:
        """Returns an LLVM :code:`opt` command line invocation for the given actions.

        :param actions: A list of actions to serialize.

        :returns: A command line string.
        """
        return f"opt {self.wrapped.to_string(actions)} input.bc -o output.bc"

    def from_string(self, string: str) -> List[ActionType]:
        """Returns a list of actions from the given command line.

        :param commandline: A command line invocation.

        :return: A list of actions.

        :raises ValueError: In case the command line string is malformed.
        """
        if string.startswith("opt "):
            string = string[len("opt ") :]

        if string.endswith(" input.bc -o output.bc"):
            string = string[: -len(" input.bc -o output.bc")]

        return self.wrapped.from_string(string)
