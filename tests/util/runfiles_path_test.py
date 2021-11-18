# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for compiler_gym/util/locks.py"""
from datetime import datetime
from pathlib import Path
from threading import Thread

from flaky import flaky

from compiler_gym.util.runfiles_path import create_user_logs_dir
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.common"]


@flaky  # Unlikely event that timestamps change
def test_create_user_logs_dir(temporary_environ, tmpdir):
    tmpdir = Path(tmpdir)
    temporary_environ["COMPILER_GYM_LOGS"] = str(tmpdir)

    dir = create_user_logs_dir("foo")
    now = datetime.now()

    assert dir.parent.parent == tmpdir / "foo"

    year, month, day = dir.parent.name.split("-")
    assert int(year) == now.year
    assert int(month) == now.month
    assert int(day) == now.day

    hour, minute, second = dir.name.split("-")
    assert int(hour) == now.hour
    assert int(minute) == now.minute
    assert int(second) == now.second


def test_create_user_logs_dir_multithreaded(temporary_environ, tmpdir):
    tmpdir = Path(tmpdir)
    temporary_environ["COMPILER_GYM_LOGS"] = str(tmpdir)

    class MakeDir(Thread):
        def __init__(self):
            super().__init__()
            self.dir = None

        def run(self):
            self.dir = create_user_logs_dir("foo")

        def join(self):
            super().join()
            return self.dir

    threads = [MakeDir() for _ in range(5)]
    for t in threads:
        t.start()

    dirs = [t.join() for t in threads]

    # Every directory should be unique.
    print(dirs)
    assert len(set(dirs)) == len(dirs)


if __name__ == "__main__":
    main()
