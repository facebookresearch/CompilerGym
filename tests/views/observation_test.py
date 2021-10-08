# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/views."""
import numpy as np
import pytest

from compiler_gym.service.connection import ServiceError
from compiler_gym.service.proto import (
    ObservationSpace,
    ScalarLimit,
    ScalarRange,
    ScalarRangeList,
)
from compiler_gym.views import ObservationView
from tests.test_main import main


class MockRawStep:
    """Mock for the raw_step callack of ObservationView."""

    def __init__(self, ret=None):
        self.called_observation_spaces = []
        self.ret = list(reversed(ret or [None]))

    def __call__(self, actions, observations, rewards):
        assert not actions
        assert len(observations) == 1
        assert not rewards
        self.called_observation_spaces.append(observations[0].id)
        ret = self.ret[-1]
        del self.ret[-1]
        return [ret], [], False, {}


def test_empty_space():
    with pytest.raises(ValueError) as ctx:
        ObservationView(MockRawStep(), [])
    assert str(ctx.value) == "No observation spaces"


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
    mock = MockRawStep(
        ret=[
            "Hello, IR",
            [1.0, 2.0],
            [-5, 15],
            b"Hello, bytes\0",
            "Hello, IR",
            [1.0, 2.0],
            [-5, 15],
            b"Hello, bytes\0",
        ]
    )
    observation = ObservationView(mock, spaces)

    value = observation["ir"]
    assert isinstance(value, str)
    assert value == "Hello, IR"

    value = observation["dfeat"]
    np.testing.assert_array_almost_equal(value, [1.0, 2.0])

    value = observation["features"]
    np.testing.assert_array_equal(value, [-5, 15])

    value = observation["binary"]
    assert value == b"Hello, bytes\0"

    # Check that the correct observation_space_list indices were used.
    assert mock.called_observation_spaces == ["ir", "dfeat", "features", "binary"]
    mock.called_observation_spaces = []

    # Repeat the above tests using the generated bound methods.
    value = observation.ir()
    assert isinstance(value, str)
    assert value == "Hello, IR"

    value = observation.dfeat()
    np.testing.assert_array_almost_equal(value, [1.0, 2.0])

    value = observation.features()
    np.testing.assert_array_equal(value, [-5, 15])

    value = observation.binary()
    assert value == b"Hello, bytes\0"

    # Check that the correct observation_space_list indices were used.
    assert mock.called_observation_spaces == ["ir", "dfeat", "features", "binary"]


def test_observation_when_raw_step_returns_incorrect_no_of_observations():
    """Test that a ServiceError is propagated when raw_step() returns unexpected
    number of observations."""

    def make_failing_raw_step(n: int):
        def failing_raw_step(*args, **kwargs):
            """A callback that returns done=True."""
            del args  # Unused
            del kwargs  # Unused
            return ["ir"] * n, None, False, {}

        return failing_raw_step

    spaces = [
        ObservationSpace(
            name="ir",
            string_size_range=ScalarRange(min=ScalarLimit(value=0)),
        )
    ]

    observation = ObservationView(make_failing_raw_step(0), spaces)
    with pytest.raises(
        ServiceError, match=r"^Expected 1 'ir' observation but the service returned 0$"
    ):
        observation["ir"]

    observation = ObservationView(make_failing_raw_step(3), spaces)
    with pytest.raises(
        ServiceError, match=r"^Expected 1 'ir' observation but the service returned 3$"
    ):
        observation["ir"]


def test_observation_when_raw_step_returns_done():
    """Test that a SessionNotFoundError from the raw_step() callback propagates as a """

    def make_failing_raw_step(error_msg=None):
        def failing_raw_step(*args, **kwargs):
            """A callback that returns done=True."""
            info = {}
            if error_msg:
                info["error_details"] = error_msg
            return [], None, True, info

        return failing_raw_step

    spaces = [
        ObservationSpace(
            name="ir",
            string_size_range=ScalarRange(min=ScalarLimit(value=0)),
        )
    ]

    observation = ObservationView(make_failing_raw_step(), spaces)
    with pytest.raises(ServiceError, match=r"^Failed to compute observation 'ir'$"):
        observation["ir"]  # pylint: disable=pointless-statement

    observation = ObservationView(make_failing_raw_step("Oh no!"), spaces)
    with pytest.raises(
        ServiceError, match=r"^Failed to compute observation 'ir': Oh no!$"
    ):
        observation["ir"]  # pylint: disable=pointless-statement


if __name__ == "__main__":
    main()
