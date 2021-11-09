# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for compiler_gym.util.executor."""
import sys
from typing import Iterable

import pytest

from compiler_gym.util.executor import Executor


def submitit_installed():
    """Determine if submitit library is available."""
    try:
        import submitit  # noqa

        return True
    except ImportError:
        return False


def executor_types() -> Iterable[str]:
    """Yield the types of executor."""
    yield "local"
    yield "debug"
    if submitit_installed():
        yield "slurm"


@pytest.fixture(scope="module", params=list(executor_types()))
def executor_type(request) -> str:
    """Test fixture which yields an executor type."""
    return request.param


def _hello_fn():
    return "Hello, world"


@pytest.mark.xfail(
    sys.platform == "darwin",
    reason="'ResourceWarning: unclosed <socket.socket ...>' when type == local",
)
def test_no_args_call(tmpdir, executor_type: str):
    with Executor(type=executor_type, cpus=1).get_executor(logs_dir=tmpdir) as executor:
        job = executor.submit(_hello_fn)
        assert job.result() == "Hello, world"


def _add_fn(a, b, *args, **kwargs):
    return a + b + sum(args) + kwargs["c"]


def test_call_with_args(tmpdir, executor_type: str):
    with Executor(type=executor_type, cpus=1).get_executor(logs_dir=tmpdir) as executor:
        job = executor.submit(_add_fn, 1, 1, 1, 1, c=1, d=None)
        assert job.result() == 5
