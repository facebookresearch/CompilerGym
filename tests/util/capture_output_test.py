# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/util:capture_output."""
import sys

from compiler_gym.util.capture_output import capture_output
from tests.test_main import main


def test_capture_print_statements():
    with capture_output() as out:
        print("Hello")
        print("World!", file=sys.stderr)

    assert out.stdout == "Hello\n"
    assert out.stderr == "World!\n"


def test_nested_capture():
    with capture_output() as outer:
        with capture_output() as inner:
            print("Hello")
        print("World!")

    assert inner.stdout == "Hello\n"
    assert outer.stdout == "World!\n"


if __name__ == "__main__":
    main()
