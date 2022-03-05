# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections.abc import Iterable as IterableType
from typing import Dict, Iterable, List, Optional, Union

from compiler_gym.envs import CompilerEnv
from compiler_gym.spaces import Commandline, CommandlineFlag, Reward
from compiler_gym.util.gym_type_hints import StepType
from compiler_gym.views import ObservationSpaceSpec
from compiler_gym.wrappers.core import ActionWrapper, CompilerEnvWrapper


class CommandlineWithTerminalAction(CompilerEnvWrapper):
    """Creates a new action space with a special "end of episode" terminal
    action at the start. If step() is called with it, the "done" flag is set.
    """

    def __init__(
        self,
        env: CompilerEnv,
        terminal=CommandlineFlag(
            name="end-of-episode",
            flag="# end-of-episode",
            description="End the episode",
        ),
    ):
        """Constructor.

        :param env: The environment to wrap.

        :param terminal: The flag to use as the terminal action. Optional.
        """
        super().__init__(env)

        if not isinstance(env.action_space, Commandline):
            raise TypeError(
                f"Unsupported action space: {type(env.action_space).__name__}"
            )

        # Redefine the action space, inserting the terminal action at the start.
        self.action_space = Commandline(
            items=[terminal]
            + [
                CommandlineFlag(
                    name=name,
                    flag=flag,
                    description=description,
                )
                for name, flag, description in zip(
                    env.action_space.names,
                    env.action_space.flags,
                    env.action_space.descriptions,
                )
            ],
            name=f"{type(self).__name__}<{env.action_space.name}>",
        )

    def step(self, action: int) -> StepType:
        end_of_episode = action == 0
        if end_of_episode:
            if self.observation_space_spec:
                observation_spaces: List[ObservationSpaceSpec] = [
                    self.observation_space_spec
                ]
            else:
                observation_spaces: List[ObservationSpaceSpec] = []
            if self.reward_space:
                reward_spaces: List[Reward] = [self.reward_space]
            else:
                reward_spaces: List[Reward] = []
            observation, reward, done, info = self.env.raw_step(
                [], observation_spaces, reward_spaces
            )
            if not done:
                done = True
                info["terminal_action"] = True
        else:
            observation, reward, done, info = self.env.step(action - 1)
        return observation, reward, done, info


class ConstrainedCommandline(ActionWrapper):
    """Constrains a Commandline action space to a subset of the original space's
    flags.
    """

    def __init__(
        self, env: CompilerEnv, flags: Iterable[str], name: Optional[str] = None
    ):
        """Constructor.

        :param env: The environment to wrap.

        :param flags: A list of entries from :code:`env.action_space.flags`
            denoting flags that are available in this wrapped environment.

        :param name: The name of the new action space.
        """
        super().__init__(env)
        self._flags = flags

        if not flags:
            raise TypeError("No flags provided")
        if not isinstance(env.action_space, Commandline):
            raise TypeError(
                "Can only wrap Commandline action space. "
                f"Received: {type(env.action_space).__name__}"
            )

        self._forward_translation: List[int] = [self.action_space[f] for f in flags]
        self._reverse_translation: Dict[int, int] = {
            v: i for i, v in enumerate(self._forward_translation)
        }

        # Redefine the action space using this smaller set of flags.
        self.action_space = Commandline(
            items=[
                CommandlineFlag(
                    name=env.action_space.names[a],
                    flag=env.action_space.flags[a],
                    description=env.action_space.descriptions[a],
                )
                for a in (env.action_space.flags.index(f) for f in flags)
            ],
            name=f"{type(self).__name__}<{name or env.action_space.name}, {len(flags)}>",
        )

    def action(self, action: Union[int, List[int]]):
        if isinstance(action, IterableType):
            return [self._forward_translation[a] for a in action]
        return self._forward_translation[action]

    def reverse_action(self, action: Union[int, List[int]]):
        if isinstance(action, IterableType):
            return [self._reverse_translation[a] for a in action]
        return self._reverse_translation[action]

    @property
    def actions(self) -> List[int]:
        """Reverse-translate actions back into the constrained space."""
        return self.reverse_action(self.env.actions)

    def fork(self) -> "ConstrainedCommandline":
        return ConstrainedCommandline(
            env=self.env.fork(), flags=self._flags, name=self.action_space.name
        )
