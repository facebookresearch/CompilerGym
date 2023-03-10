# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/spaces:scalar."""
from compiler_gym.spaces import Commandline, CommandlineFlag
from tests.test_main import main


def test_sample():
    space = Commandline(
        [
            CommandlineFlag(name="a", flag="-a", description=""),
            CommandlineFlag(name="b", flag="-b", description=""),
            CommandlineFlag(name="c", flag="-c", description=""),
        ],
        name="test",
    )
    assert space.sample() in {0, 1, 2}


def test_contains():
    space = Commandline(
        [
            CommandlineFlag(name="a", flag="-a", description=""),
            CommandlineFlag(name="b", flag="-b", description=""),
            CommandlineFlag(name="c", flag="-c", description=""),
        ],
        name="test",
    )
    assert space.contains(0)
    assert space.contains(1)
    assert space.contains(2)
    assert not space.contains(-11)
    assert not space.contains(1.5)
    assert not space.contains(4)


def test_to_and_from_string():
    space = Commandline(
        [
            CommandlineFlag(name="a", flag="-a", description=""),
            CommandlineFlag(name="b", flag="-b", description=""),
            CommandlineFlag(name="c", flag="-c", description=""),
        ],
        name="test",
    )

    assert space.to_string([0, 1, 2]) == "-a -b -c"
    assert space.from_string(space.to_string([0, 1, 2])) == [0, 1, 2]


if __name__ == "__main__":
    main()
