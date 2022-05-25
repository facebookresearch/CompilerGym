# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/util:filesystem."""
from pathlib import Path

import pytest

from compiler_gym.util import filesystem
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.common"]


def test_atomic_file_write_path(tmpwd: Path):
    out = Path("a").resolve()

    assert not out.is_file()

    with filesystem.atomic_file_write(out) as tmp_out:
        assert tmp_out != out
        assert tmp_out.parent == out.parent

        # Write to the temporary file as normal.
        with open(tmp_out, "w") as f:
            f.write("Hello!")

    with open(out) as f:
        assert f.read() == "Hello!"
    assert not tmp_out.is_file()


def test_atomic_file_write_binary_io(tmpwd: Path):
    out = Path("a").resolve()

    with filesystem.atomic_file_write(out, fileobj=True) as f:
        f.write("Hello!".encode("utf-8"))

    with open(out) as f:
        assert f.read() == "Hello!"


def test_atomic_file_write_text_io(tmpwd: Path):
    out = Path("a").resolve()

    with filesystem.atomic_file_write(out, fileobj=True, mode="w") as f:
        f.write("Hello!")

    with open(out) as f:
        assert f.read() == "Hello!"


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/dev/null",
        Path("/"),
        Path("/dev/null"),
    ],
)
def test_not_is_in_memory(path):
    assert not filesystem.is_in_memory(path)


@pytest.mark.parametrize(
    "path",
    [
        "/dev/shm",
        "/dev/shm/foo",
        Path("/dev/shm"),
        Path("/dev/shm/foo"),
    ],
)
def test_is_in_memory(path):
    assert filesystem.is_in_memory(path)


if __name__ == "__main__":
    main()
