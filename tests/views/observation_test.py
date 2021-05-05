# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/views."""
import numpy as np
import pytest
from gym.spaces import Box

from compiler_gym.service.proto import (
    DoubleList,
    Int64List,
    Observation,
    ObservationSpace,
    ScalarLimit,
    ScalarRange,
    ScalarRangeList,
    StepRequest,
)
from compiler_gym.views import ObservationView
from tests.test_main import main


class MockGetObservationReply:
    def __init__(self, value):
        self.observation = [value]


class MockGetObservation:
    """Mock for the get_observation callack of ObservationView."""

    def __init__(self, ret=None):
        self.called_observation_spaces = []
        self.ret = list(reversed(ret or []))

    def __call__(self, request: StepRequest):
        self.called_observation_spaces.append(request.observation_space[0])
        ret = self.ret[-1]
        del self.ret[-1]
        return MockGetObservationReply(ret)


def test_empty_space():
    with pytest.raises(ValueError) as ctx:
        ObservationView(MockGetObservation(), [])
    assert str(ctx.value) == "No observation spaces"


def test_invalid_observation_name():
    spaces = [
        ObservationSpace(
            name="ir",
            string_size_range=ScalarRange(min=ScalarLimit(value=0)),
        )
    ]
    observation = ObservationView(MockGetObservation(), spaces)
    with pytest.raises(KeyError) as ctx:
        _ = observation["invalid"]

    assert str(ctx.value) == "'invalid'"


def test_invalid_observation_index():
    spaces = [
        ObservationSpace(
            name="ir",
            string_size_range=ScalarRange(min=ScalarLimit(value=0)),
        )
    ]
    observation = ObservationView(MockGetObservation(), spaces)
    with pytest.raises(KeyError):
        _ = observation[100]


def test_observed_value_types():
    spaces = [
        ObservationSpace(
            name="ir",
            string_size_range=ScalarRange(min=ScalarLimit(value=0)),
        ),
        ObservationSpace(
            name="features",
            int64_range_list=ScalarRangeList(
                range=[
                    ScalarRange(
                        min=ScalarLimit(value=-100), max=ScalarLimit(value=100)
                    ),
                    ScalarRange(
                        min=ScalarLimit(value=-100), max=ScalarLimit(value=100)
                    ),
                ]
            ),
        ),
        ObservationSpace(
            name="dfeat",
            double_range_list=ScalarRangeList(
                range=[
                    ScalarRange(min=ScalarLimit(value=0.5), max=ScalarLimit(value=2.5))
                ]
            ),
        ),
        ObservationSpace(
            name="binary",
            binary_size_range=ScalarRange(
                min=ScalarLimit(value=5), max=ScalarLimit(value=5)
            ),
        ),
    ]
    mock = MockGetObservation(
        ret=[
            Observation(string_value="Hello, IR"),
            Observation(double_list=DoubleList(value=[1.0, 2.0])),
            Observation(int64_list=Int64List(value=[-5, 15])),
            Observation(binary_value=b"Hello, bytes\0"),
            Observation(string_value="Hello, IR"),
            Observation(double_list=DoubleList(value=[1.0, 2.0])),
            Observation(int64_list=Int64List(value=[-5, 15])),
            Observation(binary_value=b"Hello, bytes\0"),
        ]
    )
    observation = ObservationView(mock, spaces)

    value = observation["ir"]
    assert isinstance(value, str)
    assert value == "Hello, IR"

    value = observation["dfeat"]
    np.testing.assert_array_almost_equal(value, [1.0, 2.0])
    assert value.dtype == np.float64

    value = observation["features"]
    np.testing.assert_array_equal(value, [-5, 15])
    assert value.dtype == np.int64

    value = observation["binary"]
    assert value == b"Hello, bytes\0"

    # Check that the correct observation_space_list indices were used.
    assert mock.called_observation_spaces == [0, 2, 1, 3]
    mock.called_observation_spaces = []

    # Repeat the above tests using the generated bound methods.
    value = observation.ir()
    assert isinstance(value, str)
    assert value == "Hello, IR"

    value = observation.dfeat()
    np.testing.assert_array_almost_equal(value, [1.0, 2.0])
    assert value.dtype == np.float64

    value = observation.features()
    np.testing.assert_array_equal(value, [-5, 15])
    assert value.dtype == np.int64

    value = observation.binary()
    assert value == b"Hello, bytes\0"

    # Check that the correct observation_space_list indices were used.
    assert mock.called_observation_spaces == [0, 2, 1, 3]


def test_add_derived_space():
    spaces = [
        ObservationSpace(
            name="ir",
            string_size_range=ScalarRange(min=ScalarLimit(value=0)),
        ),
    ]
    mock = MockGetObservation(
        ret=[
            Observation(string_value="Hello, world!"),
            Observation(string_value="Hello, world!"),
        ],
    )
    observation = ObservationView(mock, spaces)
    observation.add_derived_space(
        id="ir_len",
        base_id="ir",
        space=Box(low=0, high=float("inf"), shape=(1,), dtype=int),
        translate=lambda base: [
            len(base),
        ],
    )

    value = observation["ir_len"]
    assert isinstance(value, list)
    assert value == [
        len("Hello, world!"),
    ]

    # Repeat the above test using the generated bound method.
    value = observation.ir_len()
    assert isinstance(value, list)
    assert value == [
        len("Hello, world!"),
    ]


if __name__ == "__main__":
    main()
