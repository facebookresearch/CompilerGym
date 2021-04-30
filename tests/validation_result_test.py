# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym:validation_result."""
import json

import pytest

from compiler_gym import CompilerEnvState, ValidationError, ValidationResult
from tests.test_main import main


def test_validation_error_equality():
    e1 = ValidationError(
        type="Syntax Error",
        data={"data": [1, 2, 3]},
    )
    e2 = ValidationError(  # Same as e1
        type="Syntax Error",
        data={"data": [1, 2, 3]},
    )
    e3 = ValidationError(  # Different "type"
        type="Foobar",
        data={"data": [1, 2, 3]},
    )
    e4 = ValidationError(  # Different "data" dict
        type="Syntax Error",
        data={"data": [1, 2, 3, 4]},
    )

    assert e1 == e2
    assert e1 != e3
    assert e3 != e4


def test_validation_error_json():
    """Check that JSON serialize/de-serialize produces equivalent errors."""
    error = ValidationError(
        type="Syntax Error",
        data={"data": [1, 2, 3]},
    )

    assert ValidationError(**json.loads(error.json())) == error


def test_validation_result_json():
    result = ValidationResult(
        state=CompilerEnvState(
            benchmark="benchmark://example-v0/test",
            commandline="test",
            walltime=1,
        ),
        walltime=3,
        errors=[
            ValidationError(
                type="Syntax Error",
                data={"data": [1, 2, 3]},
            )
        ],
    )

    assert ValidationResult(**json.loads(result.json())) == result


def test_validation_result_equality_different_states():
    a = ValidationResult(
        state=CompilerEnvState(
            benchmark="benchmark://example-v0/test/a",
            commandline="test",
            walltime=1,
        ),
        walltime=3,
    )
    b = ValidationResult(
        state=CompilerEnvState(
            benchmark="benchmark://example-v0/test/b",
            commandline="test",
            walltime=1,
        ),
        walltime=3,
    )
    assert a != b


def test_validation_result_equality_different_walltimes():
    a = ValidationResult(
        state=CompilerEnvState(
            benchmark="benchmark://example-v0/test",
            commandline="test",
            walltime=1,
        ),
        walltime=3,
    )
    b = ValidationResult(
        state=CompilerEnvState(
            benchmark="benchmark://example-v0/test",
            commandline="test",
            walltime=10,
        ),
        walltime=10,
    )
    assert a == b


def test_validation_result_equality_different_errors_order():
    a = ValidationResult(
        state=CompilerEnvState(
            benchmark="benchmark://example-v0/test",
            commandline="test",
            walltime=1,
        ),
        walltime=3,
        errors=[
            ValidationError(
                type="Syntax Error",
                data={"data": [1, 2, 3]},
            ),
            ValidationError(
                type="Runtime Error",
                data={"a": "b"},
            ),
        ],
    )
    b = ValidationResult(
        state=CompilerEnvState(
            benchmark="benchmark://example-v0/test",
            commandline="test",
            walltime=1,
        ),
        walltime=3,
        errors=[
            ValidationError(
                type="Runtime Error",
                data={"a": "b"},
            ),
            ValidationError(
                type="Syntax Error",
                data={"data": [1, 2, 3]},
            ),
        ],
    )
    assert a == b


def test_validation_result_join_no_inputs():
    with pytest.raises(ValueError) as e_ctx:
        ValidationResult.join([])
    assert str(e_ctx.value) == "No states to join"


def test_validation_result_join_one_input():
    result = ValidationResult(
        state=CompilerEnvState(
            benchmark="benchmark://example-v0/test",
            commandline="test",
            walltime=1,
        ),
        walltime=3,
        errors=[
            ValidationError(
                type="Syntax Error",
                data={"data": [1, 2, 3]},
            )
        ],
    )

    joined_result = ValidationResult.join([result])
    assert result == joined_result


def test_validation_result_join_two_inputs_different_errors():
    a = ValidationResult(
        state=CompilerEnvState(
            benchmark="benchmark://example-v0/test",
            commandline="test",
            walltime=1,
        ),
        walltime=3,
        errors=[
            ValidationError(
                type="Syntax Error",
                data={"data": [1, 2, 3]},
            )
        ],
    )
    b = ValidationResult(
        state=CompilerEnvState(
            benchmark="benchmark://example-v0/test",
            commandline="test",
            walltime=10,
        ),
        walltime=3,
        errors=[
            ValidationError(
                type="Type Error",
                data={"a": "b"},
            )
        ],
    )

    c = ValidationResult.join([a, b])
    assert c == ValidationResult(
        state=CompilerEnvState(
            benchmark="benchmark://example-v0/test",
            commandline="test",
            walltime=10,
        ),
        walltime=3,
        errors=[
            ValidationError(
                type="Syntax Error",
                data={"data": [1, 2, 3]},
            ),
            ValidationError(
                type="Type Error",
                data={"a": "b"},
            ),
        ],
    )
    # Test walltime, which is excluded from equality comparisons.
    assert c.walltime == 6


if __name__ == "__main__":
    main()
