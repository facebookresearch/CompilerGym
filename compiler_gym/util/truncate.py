# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def truncate(
    string: str,
    max_line_len: int = 60,
    max_lines: int = 1,
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
    if max_line_len <= 3:
        raise ValueError("Lines must be greater than 3 characeters long.")

    def _truncate_line(line: str):
        if len(line) > max_line_len:
            return f"{line[:max_line_len-3]}..."
        return line

    lines = string.split("\n")
    num_lines = len(lines)
    lines = [_truncate_line(ln) for ln in lines[:max_lines]]
    if len(lines) < num_lines:
        lines[-1] = _truncate_line(f"{lines[-1]}...")
    return "\n".join(lines)
