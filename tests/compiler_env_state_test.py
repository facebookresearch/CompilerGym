# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym:compiler_env_state."""
import json
from io import StringIO
from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

from compiler_gym import CompilerEnvState, CompilerEnvStateWriter
from compiler_gym.compiler_env_state import CompilerEnvStateReader
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.common"]


def test_state_from_dict_empty():
    with pytest.raises(PydanticValidationError):
        CompilerEnvState(**{})


def test_state_invalid_benchmark_uri():
    with pytest.raises(PydanticValidationError, match="benchmark"):
        CompilerEnvState(benchmark="invalid", walltime=100, reward=1.5, commandline="")


def test_state_invalid_walltime():
    with pytest.raises(PydanticValidationError, match="Walltime cannot be negative"):
        CompilerEnvState(
            benchmark="benchmark://cbench-v0/foo",
            walltime=-1,
            reward=1.5,
            commandline="",
        )


def test_state_to_json_from_dict():
    original_state = CompilerEnvState(
        benchmark="benchmark://cbench-v0/foo",
        walltime=100,
        reward=1.5,
        commandline="-a -b -c",
    )
    state_from_dict = CompilerEnvState(**json.loads(original_state.json()))

    assert state_from_dict.benchmark == "benchmark://cbench-v0/foo"
    assert state_from_dict.walltime == 100
    assert state_from_dict.reward == 1.5
    assert state_from_dict.commandline == "-a -b -c"


def test_state_to_json_from_dict_no_reward():
    original_state = CompilerEnvState(
        benchmark="benchmark://cbench-v0/foo", walltime=100, commandline="-a -b -c"
    )
    state_from_dict = CompilerEnvState(**json.loads(original_state.json()))

    assert state_from_dict.benchmark == "benchmark://cbench-v0/foo"
    assert state_from_dict.walltime == 100
    assert state_from_dict.reward is None
    assert state_from_dict.commandline == "-a -b -c"


def test_state_equality_different_types():
    state = CompilerEnvState(
        benchmark="benchmark://cbench-v0/foo", walltime=10, commandline="-a -b -c"
    )
    assert not state == 5  # noqa testing __eq__
    assert state != 5  # testing __ne__


def test_state_equality_same():
    a = CompilerEnvState(
        benchmark="benchmark://cbench-v0/foo", walltime=10, commandline="-a -b -c"
    )
    b = CompilerEnvState(
        benchmark="benchmark://cbench-v0/foo", walltime=10, commandline="-a -b -c"
    )
    assert a == b  # testing __eq__
    assert not a != b  # noqa testing __ne__


def test_state_equality_differnt_walltime():
    """Test that walltime is not compared."""
    a = CompilerEnvState(
        benchmark="benchmark://cbench-v0/foo", walltime=10, commandline="-a -b -c"
    )
    b = CompilerEnvState(
        benchmark="benchmark://cbench-v0/foo", walltime=5, commandline="-a -b -c"
    )
    assert a == b  # testing __eq__
    assert not a != b  # noqa testing __ne__


def test_state_equality_one_sided_reward():
    a = CompilerEnvState(
        benchmark="benchmark://cbench-v0/foo",
        walltime=5,
        commandline="-a -b -c",
        reward=2,
    )
    b = CompilerEnvState(
        benchmark="benchmark://cbench-v0/foo", walltime=5, commandline="-a -b -c"
    )
    assert a == b  # testing __eq__
    assert b == a  # testing __eq__
    assert not a != b  # noqa testing __ne__
    assert not b != a  # noqa testing __ne__


def test_state_equality_equal_reward():
    a = CompilerEnvState(
        benchmark="benchmark://cbench-v0/foo",
        walltime=5,
        commandline="-a -b -c",
        reward=2,
    )
    b = CompilerEnvState(
        benchmark="benchmark://cbench-v0/foo",
        walltime=5,
        commandline="-a -b -c",
        reward=2,
    )
    assert a == b  # testing __eq__
    assert b == a  # testing __eq__
    assert not a != b  # noqa testing __ne__
    assert not b != a  # noqa testing __ne__


def test_state_equality_unequal_reward():
    a = CompilerEnvState(
        benchmark="benchmark://cbench-v0/foo",
        walltime=5,
        commandline="-a -b -c",
        reward=2,
    )
    b = CompilerEnvState(
        benchmark="benchmark://cbench-v0/foo",
        walltime=5,
        commandline="-a -b -c",
        reward=3,
    )
    assert not a == b  # noqa testing __eq__
    assert not b == a  # noqatesting __eq__
    assert a != b  # testing __ne__
    assert b != a  # testing __ne__


def test_compiler_env_state_writer():
    buf = StringIO()
    writer = CompilerEnvStateWriter(buf)

    writer.write_state(
        CompilerEnvState(
            benchmark="benchmark://cbench-v0/foo",
            walltime=5,
            commandline="-a -b -c",
            reward=2,
        ),
        flush=True,
    )
    assert buf.getvalue() == (
        "benchmark,reward,walltime,commandline\n"
        "benchmark://cbench-v0/foo,2.0,5.0,-a -b -c\n"
    )


def test_compiler_env_state_writer_no_header():
    buf = StringIO()
    writer = CompilerEnvStateWriter(buf, header=False)

    writer.write_state(
        CompilerEnvState(
            benchmark="benchmark://cbench-v0/foo",
            walltime=5,
            commandline="-a -b -c",
            reward=2,
        ),
        flush=True,
    )
    assert buf.getvalue() == "benchmark://cbench-v0/foo,2.0,5.0,-a -b -c\n"


@pytest.mark.parametrize("flush", range(1))
def test_compiler_env_state_writer_with_statement(tmpwd: Path, flush: bool):
    path = Path("results.csv")
    assert not path.is_file()  # Sanity check.

    f = open(path, "w")
    with CompilerEnvStateWriter(f) as writer:
        writer.write_state(
            CompilerEnvState(
                benchmark="benchmark://cbench-v0/foo",
                walltime=5,
                commandline="-a -b -c",
                reward=2,
            ),
            flush=flush,
        )

    assert f.closed
    with open(path) as f:
        assert f.read() == (
            "benchmark,reward,walltime,commandline\n"
            "benchmark://cbench-v0/foo,2.0,5.0,-a -b -c\n"
        )


def test_compiler_env_state_reader():
    buf = StringIO(
        "benchmark,reward,walltime,commandline\n"
        "benchmark://cbench-v0/foo,2.0,5.0,-a -b -c\n"
    )

    reader = CompilerEnvStateReader(buf)

    assert list(reader) == [
        CompilerEnvState(
            benchmark="benchmark://cbench-v0/foo",
            walltime=5,
            commandline="-a -b -c",
            reward=2,
        )
    ]


def test_compiler_env_state_reader_no_header():
    buf = StringIO("benchmark://cbench-v0/foo,2.0,5.0,-a -b -c\n")
    reader = CompilerEnvStateReader(buf)
    assert list(reader) == [
        CompilerEnvState(
            benchmark="benchmark://cbench-v0/foo",
            walltime=5,
            commandline="-a -b -c",
            reward=2,
        )
    ]


def test_compiler_env_state_reader_with_header():
    buf = StringIO(
        "benchmark,reward,walltime,commandline\n"
        "benchmark://cbench-v0/foo,2.0,5.0,-a -b -c\n"
    )
    reader = CompilerEnvStateReader(buf)
    assert list(reader) == [
        CompilerEnvState(
            benchmark="benchmark://cbench-v0/foo",
            walltime=5,
            commandline="-a -b -c",
            reward=2,
        )
    ]


def test_compiler_env_state_reader_with_header_out_of_order_columns():
    buf = StringIO(
        "commandline,reward,benchmark,walltime\n"
        "-a -b -c,2.0,benchmark://cbench-v0/foo,5.0\n"
    )
    reader = CompilerEnvStateReader(buf)
    assert list(reader) == [
        CompilerEnvState(
            benchmark="benchmark://cbench-v0/foo",
            walltime=5,
            commandline="-a -b -c",
            reward=2,
        )
    ]


def test_compiler_env_state_reader_empty_input():
    buf = StringIO("")
    reader = CompilerEnvStateReader(buf)
    assert list(reader) == []


def test_compiler_env_state_reader_header_only():
    buf = StringIO("benchmark,reward,walltime,commandline\n")
    reader = CompilerEnvStateReader(buf)
    assert list(reader) == []


def test_state_from_csv_invalid_format():
    buf = StringIO("abcdef\n")
    reader = CompilerEnvStateReader(buf)
    with pytest.raises(
        ValueError, match=r"Expected 4 columns in the first row of CSV: \['abcdef'\]"
    ):
        next(iter(reader))


def test_state_from_csv_invalid_benchmark_uri():
    buf = StringIO(
        "benchmark,reward,walltime,commandline\n" "invalid-uri,2.0,5.0,-a -b -c\n"
    )
    reader = CompilerEnvStateReader(buf)
    with pytest.raises(ValueError, match="string does not match regex"):
        next(iter(reader))


def test_state_serialize_deserialize_equality():
    original_state = CompilerEnvState(
        benchmark="benchmark://cbench-v0/foo",
        walltime=100,
        reward=1.5,
        commandline="-a -b -c",
    )
    buf = StringIO()
    CompilerEnvStateWriter(buf).write_state(original_state)
    buf.seek(0)  # Rewind the buffer for reading.
    state_from_csv = next(iter(CompilerEnvStateReader(buf)))

    assert state_from_csv.benchmark == "benchmark://cbench-v0/foo"
    assert state_from_csv.walltime == 100
    assert state_from_csv.reward == 1.5
    assert state_from_csv.commandline == "-a -b -c"


def test_state_serialize_deserialize_equality_no_reward():
    original_state = CompilerEnvState(
        benchmark="benchmark://cbench-v0/foo", walltime=100, commandline="-a -b -c"
    )
    buf = StringIO()
    CompilerEnvStateWriter(buf).write_state(original_state)
    buf.seek(0)  # Rewind the buffer for reading.
    state_from_csv = next(iter(CompilerEnvStateReader(buf)))

    assert state_from_csv.benchmark == "benchmark://cbench-v0/foo"
    assert state_from_csv.walltime == 100
    assert state_from_csv.reward is None
    assert state_from_csv.commandline == "-a -b -c"


if __name__ == "__main__":
    main()
