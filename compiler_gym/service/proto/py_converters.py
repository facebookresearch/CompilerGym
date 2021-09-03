# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Callable, List, NamedTuple

import gym
from gym.spaces import Dict, Discrete, Tuple

from compiler_gym.service.proto.compiler_gym_service_pb2 import (
    Action,
    ActionSpace,
    Choice,
    ChoiceSpace,
)
from compiler_gym.spaces.commandline import Commandline, CommandlineFlag
from compiler_gym.spaces.named_discrete import NamedDiscrete
from compiler_gym.spaces.scalar import Scalar


class PyChoiceSpace(NamedTuple):
    """An choice space with a callback to construction Choice messages."""

    space: gym.Space
    make_choice: Callable[[Any], Choice]


class PyActionSpace(NamedTuple):
    """An action space with a callback to construction Action messages."""

    space: gym.Space
    make_action: Callable[[Any], Action]


def compose_choices(choices: List[PyChoiceSpace]) -> Callable[[Any], Action]:
    """Compose a list of choice spaces into a single make_action() callback."""

    def _compose_choices(action: Any) -> Action:
        return Action(
            choice=[choice.make_choice(a) for a, choice in zip(action, choices)]
        )

    return _compose_choices


def action_space_from_proto(proto: ActionSpace) -> PyActionSpace:
    """Construct an ActionSpace from protocol buffer."""

    def make_space_from_choice(choice: ChoiceSpace) -> PyChoiceSpace:
        choice_type = choice.WhichOneof("space")
        if choice_type == "int64_range":
            if choice.int64_range.min.value:
                return PyChoiceSpace(
                    space=Scalar(
                        min=choice.int64_range.min.value,
                        max=choice.int64_range.max.value,
                        dtype=int,
                    ),
                    make_choice=lambda a: Choice(int64_value=a),
                )
            else:
                return PyChoiceSpace(
                    space=Discrete(n=choice.int64_range.max.value),
                    make_choice=lambda a: Choice(named_discrete_value_index=a),
                )
        elif choice_type == "double_range":
            return PyChoiceSpace(
                space=Scalar(
                    min=choice.int64_range.min.value,
                    max=choice.int64_range.max.value,
                    dtype=float,
                ),
                make_choice=lambda a: Choice(double_value=a),
            )
        elif (
            choice_type == "named_discrete_space"
            and choice.named_discrete_space.is_commandline
        ):
            return PyChoiceSpace(
                space=Commandline(
                    # TODO: Decide whether commandline makes sense ...
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
        raise ValueError(f"Unknown space for ChoiceSpace: {choice}")

    if len(proto.choice) > 1 and proto.named_choices:
        choices = [make_space_from_choice(choice) for choice in proto.choice]
        return PyActionSpace(
            space=Dict(
                spaces={
                    choice.name: space.space
                    for choice, space in zip(proto.choice, choices)
                }
            ),
            make_action=compose_choices(choices),
        )

    elif len(proto.choice) > 1:
        choices = [make_space_from_choice(choice) for choice in proto.choice]

        return PyActionSpace(
            space=Tuple(spaces=[choice.space for choice in choices]),
            make_action=compose_choices(choices),
        )
    elif proto.choice:
        space, make_choice = make_space_from_choice(proto.choice[0])
        # When there is only a single choice, use the name of the parent space.
        space.name = proto.name
        return PyActionSpace(
            space=space, make_action=lambda a: Action(choice=[make_choice(a)])
        )
    raise ValueError("No choices set for ActionSpace")
