# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym:validate."""
import pytest
from gym.spaces import Dict, Tuple

from compiler_gym.service.proto import (
    ActionSpace,
    ChoiceSpace,
    NamedDiscreteSpace,
    ScalarLimit,
    ScalarRange,
    py_converters,
)
from compiler_gym.spaces import Commandline, Discrete, NamedDiscrete, Scalar
from tests.test_main import main


def test_proto_to_action_space_no_choices():
    with pytest.raises(ValueError, match=r"^No choices set for ActionSpace$"):
        py_converters.proto_to_action_space(ActionSpace(name="test", choice=[]))


def test_proto_to_action_space_empty_choice():
    with pytest.raises(ValueError, match=r'^Invalid ChoiceSpace: name: "invalid"'):
        py_converters.proto_to_action_space(
            ActionSpace(name="test", choice=[ChoiceSpace(name="invalid")])
        )


def test_proto_to_action_space_bounded_scalar_int_choice():
    space, to_action = py_converters.proto_to_action_space(
        ActionSpace(
            name="test",
            choice=[
                ChoiceSpace(
                    name="unnamed",
                    int64_range=ScalarRange(
                        min=ScalarLimit(value=-1), max=ScalarLimit(value=1)
                    ),
                )
            ],
        )
    )

    assert isinstance(space, Scalar)
    assert space.name == "test"
    assert space.contains(1)
    assert space.contains(0)
    assert space.contains(-1)

    assert not space.contains(2)
    assert not space.contains(-2)
    assert not space.contains("abc")
    assert not space.contains(1.5)

    action = to_action(1)
    assert len(action.choice) == 1
    assert action.choice[0].int64_value == 1

    action = to_action(0)
    assert len(action.choice) == 1
    assert action.choice[0].int64_value == 0

    with pytest.raises(TypeError):
        to_action(1.5)

    with pytest.raises(TypeError):
        to_action("abc")


def test_proto_to_action_space_discrete_choice():
    space, to_action = py_converters.proto_to_action_space(
        ActionSpace(
            name="test",
            choice=[
                ChoiceSpace(
                    name="unnamed",
                    int64_range=ScalarRange(max=ScalarLimit(value=2)),
                )
            ],
        )
    )

    assert isinstance(space, Discrete)
    assert space.name == "test"
    assert space.contains(1)
    assert space.contains(0)

    assert not space.contains(2)
    assert not space.contains(-2)
    assert not space.contains("abc")
    assert not space.contains(1.5)

    action = to_action(1)
    assert len(action.choice) == 1
    assert action.choice[0].int64_value == 1

    action = to_action(0)
    assert len(action.choice) == 1
    assert action.choice[0].int64_value == 0

    with pytest.raises(TypeError):
        to_action(1.5)

    with pytest.raises(TypeError):
        to_action("abc")


def test_proto_to_action_space_bounded_scalar_double_choice():
    space, to_action = py_converters.proto_to_action_space(
        ActionSpace(
            name="test",
            choice=[
                ChoiceSpace(
                    name="unnamed",
                    double_range=ScalarRange(
                        min=ScalarLimit(value=-1), max=ScalarLimit(value=1)
                    ),
                )
            ],
        )
    )

    assert isinstance(space, Scalar)
    assert space.name == "test"
    assert space.contains(1.0)
    assert space.contains(0.0)
    assert space.contains(-1.0)

    assert not space.contains(2.0)
    assert not space.contains(-2.0)
    assert not space.contains("abc")
    assert not space.contains(1)

    action = to_action(1)
    assert len(action.choice) == 1
    assert action.choice[0].double_value == 1

    action = to_action(0.5)
    assert len(action.choice) == 1
    assert action.choice[0].double_value == 0.5

    with pytest.raises(TypeError):
        to_action("abc")


def test_proto_to_action_space_named_discrete_choice():
    space, to_action = py_converters.proto_to_action_space(
        ActionSpace(
            name="test",
            choice=[
                ChoiceSpace(
                    name="unnamed",
                    named_discrete_space=NamedDiscreteSpace(value=["a", "b", "c"]),
                )
            ],
        )
    )

    assert isinstance(space, NamedDiscrete)
    assert space.name == "test"
    assert space.contains(0)
    assert space.contains(1)
    assert space.contains(2)

    assert not space.contains(-1)
    assert not space.contains(3)
    assert not space.contains("a")
    assert not space.contains(1.5)

    action = to_action(1)
    assert len(action.choice) == 1
    assert action.choice[0].named_discrete_value_index == 1

    action = to_action(2)
    assert len(action.choice) == 1
    assert action.choice[0].named_discrete_value_index == 2

    with pytest.raises(TypeError):
        to_action(1.5)

    with pytest.raises(TypeError):
        to_action("abc")


def test_proto_to_action_space_commandline():
    space, to_action = py_converters.proto_to_action_space(
        ActionSpace(
            name="test",
            choice=[
                ChoiceSpace(
                    name="unnamed",
                    named_discrete_space=NamedDiscreteSpace(
                        value=["a", "b", "c"], is_commandline=True
                    ),
                )
            ],
        )
    )

    assert isinstance(space, Commandline)
    assert space.name == "test"
    assert space.contains(0)
    assert space.contains(1)
    assert space.contains(2)

    assert not space.contains(-1)
    assert not space.contains(3)
    assert not space.contains("a")
    assert not space.contains(1.5)

    action = to_action(1)
    assert len(action.choice) == 1
    assert action.choice[0].named_discrete_value_index == 1

    action = to_action(2)
    assert len(action.choice) == 1
    assert action.choice[0].named_discrete_value_index == 2

    with pytest.raises(TypeError):
        to_action(2.5)

    with pytest.raises(TypeError):
        to_action("abc")


def test_proto_to_action_space_tuple_int_choices():
    space, to_action = py_converters.proto_to_action_space(
        ActionSpace(
            name="test",
            choice=[
                ChoiceSpace(
                    name="a",
                    int64_range=ScalarRange(
                        min=ScalarLimit(value=-1), max=ScalarLimit(value=1)
                    ),
                ),
                ChoiceSpace(
                    name="b",
                    int64_range=ScalarRange(
                        min=ScalarLimit(value=0), max=ScalarLimit(value=2)
                    ),
                ),
            ],
        )
    )

    assert isinstance(space, Tuple)
    assert space.name == "test"

    assert len(space.spaces) == 2
    assert isinstance(space.spaces[0], Scalar)
    assert isinstance(space.spaces[1], Discrete)

    assert space.contains((1, 0))
    assert space.contains((-1, 1))

    assert not space.contains((2, 0))
    assert not space.contains((-2, 0))
    assert not space.contains(2)
    assert not space.contains("abc")
    assert not space.contains(1.5)

    action = to_action((1, 0))
    assert len(action.choice) == 2
    assert action.choice[0].int64_value == 1
    assert action.choice[1].int64_value == 0

    with pytest.raises(TypeError):
        to_action(1)

    with pytest.raises(TypeError):
        to_action("abs")


def test_proto_to_action_space_dict_int_choices():
    space, to_action = py_converters.proto_to_action_space(
        ActionSpace(
            name="test",
            choice=[
                ChoiceSpace(
                    name="a",
                    int64_range=ScalarRange(
                        min=ScalarLimit(value=-1), max=ScalarLimit(value=1)
                    ),
                ),
                ChoiceSpace(
                    name="b",
                    int64_range=ScalarRange(
                        min=ScalarLimit(value=0), max=ScalarLimit(value=2)
                    ),
                ),
            ],
            named_choices=True,
        )
    )

    assert isinstance(space, Dict)
    assert space.name == "test"

    assert len(space.spaces) == 2
    assert isinstance(space.spaces["a"], Scalar)
    assert isinstance(space.spaces["b"], Discrete)

    assert space.contains({"a": 1, "b": 0})

    assert not space.contains({"a": 2, "b": 0})
    assert not space.contains({"a": -2, "b": 0})
    assert not space.contains((1, 0))
    assert not space.contains("ab")

    action = to_action({"a": 1, "b": 0})
    assert len(action.choice) == 2
    assert action.choice[0].int64_value == 1
    assert action.choice[1].int64_value == 0

    with pytest.raises(TypeError):
        to_action(1)

    with pytest.raises(TypeError):
        to_action("abs")


if __name__ == "__main__":
    main()
