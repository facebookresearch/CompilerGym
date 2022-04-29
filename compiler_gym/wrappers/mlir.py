# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections.abc import Mapping
from copy import deepcopy
from math import factorial
from numbers import Integral
from typing import Iterable, Optional, Union

import numpy as np
from gym.spaces import Space

from compiler_gym.envs import CompilerEnv
from compiler_gym.spaces import Box
from compiler_gym.spaces import Dict as DictSpace
from compiler_gym.spaces import (
    Discrete,
    NamedDiscrete,
    Permutation,
    Reward,
    Scalar,
    SpaceSequence,
)
from compiler_gym.spaces import Tuple as TupleSpace
from compiler_gym.util.gym_type_hints import ActionType, ObservationType, StepType
from compiler_gym.util.permutation import convert_number_to_permutation
from compiler_gym.views import ObservationSpaceSpec
from compiler_gym.wrappers.core import (
    ActionWrapper,
    CompilerEnvWrapper,
    ObservationWrapper,
)


def convert_permutation_to_discrete_space(permutation: Permutation) -> Discrete:
    return Discrete(name=permutation.name, n=factorial(permutation.size_range[0]))


def get_tile_size_discrete_space(min: Integral) -> NamedDiscrete:
    items = [str(min * 2**i) for i in range(11)]
    return NamedDiscrete(items=items, name=None)


def convert_tile_sizes_space(box: Box) -> TupleSpace:
    spaces = [get_tile_size_discrete_space(box.low[i]) for i in range(box.shape[0])]
    return TupleSpace(spaces=spaces, name=box.name)


def convert_bool_to_discrete_space(x: Scalar) -> NamedDiscrete:
    if x.min or not x.max:
        raise ValueError(
            f"Invalid scalar range [{x.min}, {x.max}. [False, True] expected."
        )
    return NamedDiscrete(name=x.name, items=["False", "True"])


def convert_action_space(
    action_space: SpaceSequence, max_subactions: Optional[Integral]
) -> Space:
    template_space = deepcopy(action_space.space)
    template_space["tile_options"][
        "interchange_vector"
    ] = convert_permutation_to_discrete_space(
        template_space["tile_options"]["interchange_vector"]
    )
    template_space["tile_options"]["tile_sizes"] = convert_tile_sizes_space(
        template_space["tile_options"]["tile_sizes"]
    )
    template_space["tile_options"]["promote"] = convert_bool_to_discrete_space(
        template_space["tile_options"]["promote"]
    )
    template_space["tile_options"][
        "promote_full_tile"
    ] = convert_bool_to_discrete_space(
        template_space["tile_options"]["promote_full_tile"]
    )
    template_space["vectorize_options"][
        "unroll_vector_transfers"
    ] = convert_bool_to_discrete_space(
        template_space["vectorize_options"]["unroll_vector_transfers"]
    )
    res = TupleSpace(name=None, spaces=[])
    for i in range(action_space.size_range[0]):
        res.spaces.append(deepcopy(template_space))
    if max_subactions is None:
        loop_bound = action_space.size_range[1]
    else:
        if action_space.size_range[0] > max_subactions:
            raise ValueError(
                f"max_subactions {max_subactions} must be greater than the minimum the environment expects {action_space.size_range[0]}."
            )
        loop_bound = max_subactions

    for i in range(action_space.size_range[0], loop_bound):
        res.spaces.append(
            DictSpace(
                name=None,
                spaces={
                    "space": deepcopy(template_space),
                    "is_present": NamedDiscrete(name=None, items=["False", "True"]),
                },
            )
        )
    return res


_tile_size_discrite_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def convert_matmul_op_action(action: ActionType) -> ActionType:
    res = deepcopy(action)
    res["tile_options"]["interchange_vector"] = convert_number_to_permutation(
        action["tile_options"]["interchange_vector"], permutation_size=3
    )
    tile_sizes = action["tile_options"]["tile_sizes"]
    res["tile_options"]["tile_sizes"] = np.array(
        [_tile_size_discrite_values[tile_sizes[i]] for i in range(len(tile_sizes))],
        dtype=int,
    )
    res["tile_options"]["promote"] = bool(action["tile_options"]["promote"])
    res["tile_options"]["promote_full_tile"] = bool(
        action["tile_options"]["promote_full_tile"]
    )
    res["vectorize_options"]["unroll_vector_transfers"] = bool(
        action["vectorize_options"]["unroll_vector_transfers"]
    )
    return res


def convert_action(action: ActionType) -> ActionType:
    res = []
    for a in action:
        if not isinstance(a, Mapping) or "is_present" not in a:
            res.append(convert_matmul_op_action(a))
        elif a["is_present"] != 0:
            res.append(convert_matmul_op_action(a["space"]))
    return res


def convert_observation_space(space: Space) -> Scalar:
    return Box(
        name=space.name,
        shape=[1],
        low=space.scalar_range.min,
        high=space.scalar_range.max,
        dtype=float,
    )


def convert_observation(observation: ObservationType) -> ObservationType:
    return (
        None if observation is None else np.array([np.median(observation)], dtype=float)
    )


class MlirRlObservationWrapperEnv(ObservationWrapper):
    @property
    def observation_space(self):
        return convert_observation_space(self.env.observation_space)

    @observation_space.setter
    def observation_space(
        self, observation_space: Optional[Union[str, ObservationSpaceSpec]]
    ) -> None:
        self.env.observation_space = observation_space

    def convert_observation(self, observation: ObservationType) -> ObservationType:
        return convert_observation(observation)


class MlirRlActionWrapperEnv(ActionWrapper):
    def __init__(
        self,
        env: CompilerEnv,
        max_subactions: Optional[Integral] = None,
    ):
        super().__init__(env)
        self.max_subactions = max_subactions

    @property
    def action_space(self) -> Space:
        return convert_action_space(
            self.env.action_space, max_subactions=self.max_subactions
        )

    @action_space.setter
    def action_space(self, action_space: Optional[str]):
        self.env.action_space = action_space

    def action(self, action: ActionType) -> ActionType:
        return convert_action(action)


class MlirRlErrorWrapperEnv(CompilerEnvWrapper):
    def multistep(
        self,
        actions: Iterable[ActionType],
        observation_spaces: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        reward_spaces: Optional[Iterable[Union[str, Reward]]] = None,
        observations: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        rewards: Optional[Iterable[Union[str, Reward]]] = None,
    ) -> StepType:
        observation, reward, done, info = super().multistep(
            actions,
            observation_spaces=observation_spaces,
            reward_spaces=reward_spaces,
            observations=observations,
            rewards=rewards,
        )
        if "error_type" in info:
            raise RuntimeError(str(info))
        return observation, reward, done, info

    def step(
        self,
        action: ActionType,
        observation_spaces: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        reward_spaces: Optional[Iterable[Union[str, Reward]]] = None,
        observations: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        rewards: Optional[Iterable[Union[str, Reward]]] = None,
    ):
        return self.multistep(
            [action], observation_spaces, reward_spaces, observations, rewards
        )


def make_mlir_rl_wrapper_env(
    env: CompilerEnv, max_subactions: Optional[Integral] = None
):
    """Create a wrapper for the MLIR environment that is suitable to interface with
    off-the-shelf RL frameworks.
    """
    env.reward_space = "runtime"
    env.observation_space = "Runtime"
    res = MlirRlActionWrapperEnv(env, max_subactions=max_subactions)
    res = MlirRlObservationWrapperEnv(res)
    res = MlirRlErrorWrapperEnv(res)
    return res
