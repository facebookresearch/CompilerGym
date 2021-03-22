# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/util:truncate."""
from compiler_gym.util.truncate import truncate, truncate_lines
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


def test_truncate_lines_no_truncation():
    assert truncate_lines(["abc"]) == "abc"
    assert (
        truncate_lines(["abcdef", "abcdef"], max_line_len=7, max_lines=2)
        == "abcdef\nabcdef"
    )


def test_truncate_lines_single_line():
    assert truncate_lines(["abcdefghijklmnop"], max_line_len=5) == "ab..."


def test_truncate_lines_dual_lines():
    assert (
        truncate_lines(
            ["abcdefghijklmnop", "bcdefghijklmnop"], max_line_len=5, max_lines=3
        )
        == "ab...\nbc..."
    )


def test_truncate_lines_dual_lines_generator():
    def gen():
        yield "abcdefghijklmnop"
        yield "bcdefghijklmnop"

    assert truncate_lines(gen(), max_line_len=5, max_lines=3) == "ab...\nbc..."


if __name__ == "__main__":
    main()
