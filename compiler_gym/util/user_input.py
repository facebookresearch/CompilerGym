# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
import termios
import tty
from typing import List

import humanize

from compiler_gym.util.shell_format import emph


def read_char() -> str:
    """Read a single character from user input. Doesn't work on Windows."""
    fd = sys.stdin.fileno()
    oldSettings = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, oldSettings)

    if ch == "\x03":  # C-c
        raise KeyboardInterrupt
    elif ch == "\x04":  # C-d
        raise EOFError
    return ch


def read_list_index(prefix: str, values: List[str], truncate_at: int = 201) -> int:
    """Prompt the user to select an element from a list and return the index."""
    index = None
    while index is None:
        print(
            f"{prefix}:",
            ", ".join(
                f"[{emph(i)}] {name}" for i, name in enumerate(values[:truncate_at])
            ),
            end="",
            flush=True,
        )
        if len(values) > truncate_at:
            print(f" (+ {humanize.intcomma(len(values) - truncate_at)} more...) ")

        if len(values) > 10:
            # If there are more elements to choose from than word fit in a
            # single digit choice, require that the user types the index or name
            # of the value.
            print(" (enter number or name) >>> ", end="", flush=True)
            line = input()
            try:
                index = int(line)
                if index >= 0 and index < len(values):
                    break
                else:
                    print("Invalid value")
                    index = None
            except ValueError:
                try:
                    # Allow the user to type the name of the item, rather than
                    # the index. E.g. '[0] foo, [1] bar >>> bar'.
                    index = values.index(line)
                    break
                except ValueError:
                    print("Invalid value")
                    index = None
        else:
            # A single 0-9 digit is sufficient, so no need to wait for return
            # key.
            print(" >>> ", end="", flush=True)
            while True:
                try:
                    index = int(read_char())
                    if index >= 0 and index < len(values):
                        print(f"{values[index]}", flush=True)
                        break
                except (ValueError, TypeError):
                    pass

    return index


def read_list_value(prefix: str, values: List[str]) -> str:
    """Prompt the user to select an element from a list and return the value."""
    return values[read_list_index(prefix, values)]
