# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for compiler_gym/util/shell_format.py"""
from compiler_gym.util import shell_format as fmt
from tests.test_main import main


def test_indent():
    assert fmt.indent("abc") == "    abc"
    assert fmt.indent("abc", n=2) == "  abc"
    assert fmt.indent("abc\ndef") == "    abc\n    def"


def test_join_cmd():
    assert fmt.join_cmd(["a", "b", "c"]) == "a b c"


if __name__ == "__main__":
    main()
