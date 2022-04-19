# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Dict, List, Optional

from llvm_autotuning.just_keep_going_env import JustKeepGoingEnv
from llvm_rl.wrappers import *  # noqa wrapper definition
from pydantic import BaseModel, Field, validator
from pydantic.class_validators import root_validator

import compiler_gym
from compiler_gym import CompilerEnv
from compiler_gym.wrappers import *  # noqa wrapper definitions
from compiler_gym.wrappers import TimeLimit


class EnvironmentWrapperConfig(BaseModel):
    """Description of a CompilerEnvWrapper class."""

    # === Start of fields list. ===

    wrapper: str = Field(allow_mutation=False)
    """The name of the wrapper class. This class name must be imported into this
    module.
    """

    args: Dict[str, Any] = Field(default={}, allow_mutation=False)
    """"A dictionary of arguments to pass to the wrapper constructor."""

    # === Start of public API. ===

    @property
    def wrapper_class(self):
        """Return the wrapper class type."""
        return self._to_class(self.wrapper)

    def wrap(self, env: CompilerEnv) -> CompilerEnv:
        """Wrap the given environment."""
        try:
            return self.wrapper_class(env=env, **self.args)
        except TypeError as e:
            raise TypeError(
                f"Error constructing CompilerEnv wrapper {self.wrapper_class.__name__}: {e}"
            ) from e

    # === Start of implementation details. ===

    @validator("wrapper")
    def validate_wrapper(cls, value):
        # Check that the class can be constructed.
        cls._to_class(value)
        return value

    @staticmethod
    def _to_class(value: str):
        try:
            return globals()[value]
        except KeyError as e:
            raise ValueError(
                f"Unknown wrapper class: {value}\n"
                "Make sure it is imported in rl/model/environment.py"
            ) from e

    class Config:
        validate_assignment = True


class Environment(BaseModel):
    """Represents a CompilerEnv environment."""

    id: str = Field(allow_mutation=False)
    """The environment ID, as passed to :code:`gym.make(...)`."""

    reward_space: Optional[str] = Field(default=None, allow_mutation=False)
    """The reward space to use, as a string."""

    observation_space: Optional[str] = Field(default=None, allow_mutation=False)
    """The observation space to use, as a string."""

    max_episode_steps: int = Field(allow_mutation=False, gt=0)
    """The maximum number of steps in an episode of this environment. For the
    sake of consistency this *must* be defined.
    """

    wrappers: List[EnvironmentWrapperConfig] = Field(default=[], allow_mutation=False)
    """A list of wrapper classes to apply to the environment."""

    rllib_id: Optional[str] = Field(allow_mutation=False)
    """The ID of the custom environment to register with RLlib. This shows up in
    the logs but has no effect on behavior. Defaults to the `id` value.
    """

    # === Start of public API. ===

    def make_env(self) -> CompilerEnv:
        """Construct a compiler environment from the given config."""
        env = compiler_gym.make(self.id)
        if self.observation_space:
            env.observation_space = self.observation_space
        if self.reward_space:
            env.reward_space = self.reward_space
        for wrapper in self.wrappers:
            env = wrapper.wrap(env)
        # Wrap the env to ignore errors during search.
        env = JustKeepGoingEnv(env)
        env = TimeLimit(env, max_episode_steps=self.max_episode_steps)
        return env

    # === Start of implementation details. ===

    @validator("id")
    def validate_id(cls, value):
        assert (
            value in compiler_gym.COMPILER_GYM_ENVS
        ), f"Not a CompilerGym environment: {value}"
        return value

    @validator("wrappers", pre=True)
    def validate_wrappers(cls, value) -> List[EnvironmentWrapperConfig]:
        # Convert the omegaconf ListConfig into a list of
        # EnvironmentWrapperConfig objects.
        return [EnvironmentWrapperConfig(**v) for v in value]

    @root_validator
    def rllib_id_default_value(cls, values):
        values["rllib_id"] = values["rllib_id"] or values["id"]
        return values

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True
