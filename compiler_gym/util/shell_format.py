# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import shlex
import sys
from typing import Any, Iterable


class ShellFormatCodes:
    """Shell escape codes for pretty-printing."""

    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def emph(stringable: Any) -> str:
    """Emphasize a string."""
    return f"{ShellFormatCodes.BOLD}{ShellFormatCodes.BLUE}{stringable}{ShellFormatCodes.END}"


def plural(quantity: int, singular: str, plural: str) -> str:
    """Return the singular or plural word."""
    return singular if quantity == 1 else plural


def indent(string: str, n=4) -> str:
    """Indent a multi-line string by given number of spaces."""
    return "\n".join(" " * n + x for x in str(string).split("\n"))


def join_cmd(cmd: Iterable[str]) -> str:
    """Join a list of command line arguments into a single string.

    This is intended for logging purposes only. It does not provide any safety
    guarantees.
    """
    if sys.version_info >= (3, 8, 0):
        return shlex.join(cmd)
    return " ".join(cmd)
