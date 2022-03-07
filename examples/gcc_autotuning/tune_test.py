# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import docker
import pytest
from absl.flags import FLAGS

from . import tune


def docker_is_available() -> bool:
    """Return whether docker is available."""
    try:
        docker.from_env()
        return True
    except docker.errors.DockerException:
        return False


def system_has_functional_gcc(gcc_path: str) -> bool:
    """Return whether there is a system GCC available."""
    try:
        stdout = subprocess.check_output(
            [gcc_path, "--version"],
            universal_newlines=True,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        # On some systems "gcc" may alias to a different compiler, so check for
        # the presence of the name "gcc" in the first line of output.
        return "gcc" in stdout.split("\n")[0].lower()
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return False


@lru_cache
def gcc_bins() -> Iterable[str]:
    """Return a list of available GCCs."""
    if docker_is_available():
        yield "docker:gcc:11.2.0"
    system_gcc = shutil.which("gcc")
    if system_gcc and system_has_functional_gcc(system_gcc):
        yield system_gcc


@pytest.fixture(scope="module", params=gcc_bins())
def gcc_bin(request) -> str:
    return request.param


@pytest.mark.parametrize("search", ["random", "hillclimb", "genetic"])
def test_tune_smoke_test(search: str, gcc_bin: str, capsys, tmpdir: Path):
    tmpdir = Path(tmpdir)
    flags = [
        "argv0",
        "--seed=0",
        f"--output_dir={tmpdir}",
        f"--gcc_bin={gcc_bin}",
        "--gcc_benchmark=benchmark://chstone-v0/aes",
        f"--search={search}",
        "--pop_size=3",
        "--gcc_search_budget=6",
    ]
    sys.argv = flags
    FLAGS.unparse_flags()
    FLAGS(flags)

    tune.main([])
    out, _ = capsys.readouterr()
    assert "benchmark://chstone-v0/aes" in out
    assert (tmpdir / "results.csv").is_file()
