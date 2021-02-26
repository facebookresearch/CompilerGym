# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/util:temporary_working_directory."""
import os
import tempfile
from pathlib import Path

from compiler_gym.util.temporary_working_directory import temporary_working_directory
from tests.test_main import main


def test_temporary_working_directory_tempdir():
    with temporary_working_directory() as cwdir:
        # Suffix test rather than equality test because on macOS temporary
        # directories can have a /private prefix.
        assert os.getcwd().endswith(str(cwdir))
        assert cwdir.is_dir()
        assert not list(cwdir.iterdir())
        (cwdir / "test").touch()
        assert (cwdir / "test").is_file()

    # Out of scope, the directory is removed.
    assert not cwdir.is_dir()


def test_temporary_working_directory():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d)
        with temporary_working_directory(path) as cwdir:
            assert path == cwdir
            # Suffix test rather than equality test because on macOS temporary
            # directories can have a /private prefix.
            assert os.getcwd().endswith(str(path))
            assert cwdir.is_dir()
            assert not list(cwdir.iterdir())
            (cwdir / "test").touch()
            assert (cwdir / "test").is_file()

        # Out of scope, the directory is preserved.
        assert path.is_dir()


if __name__ == "__main__":
    main()
