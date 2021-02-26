# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym:compiler_env_state."""
import pytest

from compiler_gym import CompilerEnvState
from tests.test_main import main


def test_state_to_csv_from_csv():
    original_state = CompilerEnvState(
        benchmark="foo", walltime=100, reward=1.5, commandline="-a -b -c"
    )
    state_from_csv = CompilerEnvState.from_csv(original_state.to_csv())

    assert state_from_csv.benchmark == "foo"
    assert state_from_csv.walltime == 100
    assert state_from_csv.reward == 1.5
    assert state_from_csv.commandline == "-a -b -c"


def test_state_to_csv_from_csv_no_reward():
    original_state = CompilerEnvState(
        benchmark="foo", walltime=100, commandline="-a -b -c"
    )
    state_from_csv = CompilerEnvState.from_csv(original_state.to_csv())

    assert state_from_csv.benchmark == "foo"
    assert state_from_csv.walltime == 100
    assert state_from_csv.reward is None
    assert state_from_csv.commandline == "-a -b -c"


def test_state_from_csv_empty():
    with pytest.raises(ValueError) as ctx:
        CompilerEnvState.from_csv("")

    assert str(ctx.value) == "Failed to parse input: ``"


def test_state_from_csv_invalid_format():
    with pytest.raises(ValueError) as ctx:
        CompilerEnvState.from_csv("abcdef")

    assert str(ctx.value).startswith("Failed to parse input: `abcdef`: ")


def test_state_to_json_from_json():
    original_state = CompilerEnvState(
        benchmark="foo", walltime=100, reward=1.5, commandline="-a -b -c"
    )
    state_from_json = CompilerEnvState.from_json(original_state.json())

    assert state_from_json.benchmark == "foo"
    assert state_from_json.walltime == 100
    assert state_from_json.reward == 1.5
    assert state_from_json.commandline == "-a -b -c"


def test_state_to_json_from_json_no_reward():
    original_state = CompilerEnvState(
        benchmark="foo", walltime=100, commandline="-a -b -c"
    )
    state_from_json = CompilerEnvState.from_json(original_state.json())

    assert state_from_json.benchmark == "foo"
    assert state_from_json.walltime == 100
    assert state_from_json.reward is None
    assert state_from_json.commandline == "-a -b -c"


def test_state_from_json_empty():
    with pytest.raises(TypeError):
        CompilerEnvState.from_json({})


if __name__ == "__main__":
    main()
