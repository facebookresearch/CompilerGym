# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/util:truncate."""
from compiler_gym.util.truncate import truncate
from tests.test_main import main


def test_truncate_no_truncation():
    assert truncate("abc") == "abc"
    assert truncate("abcdef\nabcdef", max_line_len=7, max_lines=2) == "abcdef\nabcdef"


def test_truncate_single_line():
    assert truncate("abcdefghijklmnop", max_line_len=5) == "ab..."


def test_truncate_dual_lines():
    assert (
        truncate("abcdefghijklmnop\nbcdefghijklmnop", max_line_len=5, max_lines=3)
        == "ab...\nbc..."
    )


def test_truncate_final_line():
    assert truncate("abc\ndef\n123", max_line_len=5, max_lines=2) == "abc\nde..."
    assert truncate("abc\ndef\n123", max_line_len=10, max_lines=2) == "abc\ndef..."


if __name__ == "__main__":
    main()
