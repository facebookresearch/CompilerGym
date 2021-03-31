# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
from typing import Iterable


def truncate(
    string: str,
    max_line_len: int = 60,
    max_lines: int = 1,
    tail: bool = False,
) -> str:
    """Truncate a string using ellipsis.

    For multi-line inputs, each line is truncated independently.

    For example:

    >>> truncate("abcdefghijklmnop\n1234", max_line_len=10)
    "abcdefg...\n1234"

    :param string: The string to truncate.
    :param max_line_len: The maximum number of characters in each line.
    :param max_lines: The maximum number of lines in the output string.
    :return: A (possibly truncated) string.
    """
    return truncate_lines(
        str(string).split("\n"),
        max_line_len=max_line_len,
        max_lines=max_lines,
        tail=tail,
    )


def truncate_lines(
    lines: Iterable[str],
    max_line_len: int = 60,
    max_lines: int = 1,
    tail: bool = False,
) -> str:
    """Truncate a sequence of lines, one string per line, using ellipsis.

    Each line is truncated independently and combined into a single multi-line
    string.

    For example:

    >>> truncate_lines(["abcdefghijklmnop", "1234"], max_line_len=10)
    "abcdefg...\n1234"

    :param string: The string to truncate.
    :param max_line_len: The maximum number of characters in each line.
    :param max_lines: The maximum number of lines in the output string.
    :return: A (possibly truncated) string.
    """
    if max_line_len <= 3:
        raise ValueError("Lines must be greater than 3 characeters long.")

    def _truncate_line(line: str):
        if len(line) > max_line_len:
            return f"{line[:max_line_len-3]}..."
        return line

    def _consume(iterable, n):
        """Consume fist or last `n` elements from iterable."""
        if tail:
            yield from deque(iterable, n)
        else:
            for _ in range(n):
                try:
                    yield next(iterable)
                except StopIteration:
                    return

    lines = iter(lines)

    truncated_lines = [_truncate_line(str(ln)) for ln in _consume(lines, max_lines)]

    # Truncate the final line if required.
    try:
        next(lines)
        truncated_lines[-1] = _truncate_line(f"{truncated_lines[-1]}...")
    except StopIteration:
        pass

    return "\n".join(truncated_lines)
