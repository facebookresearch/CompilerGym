# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Callable
from typing import Dict as DictType
from typing import List, NamedTuple

import gym

from compiler_gym.service.proto.compiler_gym_service_pb2 import (
    Action,
    ActionSpace,
    Choice,
    ChoiceSpace,
)
from compiler_gym.spaces.commandline import Commandline, CommandlineFlag
from compiler_gym.spaces.dict import Dict
from compiler_gym.spaces.discrete import Discrete
from compiler_gym.spaces.named_discrete import NamedDiscrete
from compiler_gym.spaces.scalar import Scalar
from compiler_gym.spaces.tuple import Tuple


class PyChoiceSpace(NamedTuple):
    """An choice space with a callback to construction Choice messages."""

    space: gym.Space
    make_choice: Callable[[Any], Choice]


class PyActionSpace(NamedTuple):
    """An action space with a callback to construction Action messages."""

    space: gym.Space
    make_action: Callable[[Any], Action]


def proto_to_action_space(proto: ActionSpace) -> PyActionSpace:
    """Convert a ActionSpace message to a gym.Space and action callback.

    :param proto: An ActionSpace message.

    :returns: A PyActionSpace tuple, comprising a gym.Space space, and a
        callback that formats its input as an Action message.
    """
    if proto.named_choices:
        # Convert a list of named choices to a dictionary space.
        choices = [proto_to_choice_space(choice) for choice in proto.choice]

        def compose_choice_dict(action: DictType[str, Any]):
            return Action(
                choice=[
                    choice.make_choice(action[choice.space.name]) for choice in choices
                ]
            )

        return PyActionSpace(
            space=Dict(
                spaces={
                    choice.name: space.space
                    for choice, space in zip(proto.choice, choices)
                },
                name=proto.name,
            ),
            make_action=compose_choice_dict,
        )
    elif len(proto.choice) > 1:
        # Convert an unnamed list of choices to a tuple space.
        choices = [proto_to_choice_space(choice) for choice in proto.choice]

        def compose_choice_list(action: List[Any]) -> Action:
            return Action(
                choice=[choice.make_choice(a) for a, choice in zip(action, choices)]
            )

        return PyActionSpace(
            space=Tuple(spaces=[choice.space for choice in choices], name=proto.name),
            make_action=compose_choice_list,
        )
    elif proto.choice:
        # Convert a single choice into an action space.
        space, make_choice = proto_to_choice_space(proto.choice[0])
        # When there is only a single choice, use the name of the parent space.
        space.name = proto.name
        return PyActionSpace(
            space=space, make_action=lambda a: Action(choice=[make_choice(a)])
        )
    raise ValueError("No choices set for ActionSpace")


def proto_to_choice_space(choice: ChoiceSpace) -> PyChoiceSpace:
    """Convert a ChoiceSpace message to a gym.Space and choice callback.

    :param proto: A ChoiceSpace message.

    :returns: A PyChoiceSpace tuple, comprising a gym.Space space, and a
        callback that wraps an input in a Choice message.
    """
    choice_type = choice.WhichOneof("space")
    if choice_type == "int64_range":
        # The Discrete class defines a discrete space as integers in the range
        # [0,n]. For spaces that aren't zero-based there is a Scalar class.
        # Prefer Discrete if possible since it is part of the core gym library,
        # else fallback to Scalar.
        if choice.int64_range.min.value:
            return PyChoiceSpace(
                space=Scalar(
                    min=choice.int64_range.min.value,
                    max=choice.int64_range.max.value,
                    dtype=int,
                    name=choice.name,
                ),
                make_choice=lambda a: Choice(int64_value=a),
            )
        else:
            return PyChoiceSpace(
                space=Discrete(n=choice.int64_range.max.value, name=choice.name),
                make_choice=lambda a: Choice(int64_value=a),
            )
    elif choice_type == "double_range":
        return PyChoiceSpace(
            space=Scalar(
                min=choice.double_range.min.value,
                max=choice.double_range.max.value,
                dtype=float,
                name=choice.name,
            ),
            make_choice=lambda a: Choice(double_value=a),
        )
    elif (
        choice_type == "named_discrete_space"
        and choice.named_discrete_space.is_commandline
    ):
        return PyChoiceSpace(
            space=Commandline(
                items=[
                    CommandlineFlag(name=x, flag=x, description="")
                    for x in choice.named_discrete_space.value
                ],
                name=choice.name,
            ),
            make_choice=lambda a: Choice(named_discrete_value_index=a),
        )
    elif choice_type == "named_discrete_space":
        return PyChoiceSpace(
            space=NamedDiscrete(
                items=choice.named_discrete_space.value, name=choice.name
            ),
            make_choice=lambda a: Choice(named_discrete_value_index=a),
        )
    raise ValueError(f"Invalid ChoiceSpace: {choice}")
