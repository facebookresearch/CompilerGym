# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any


class ShellFormatCodes(object):
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
