# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from typing import Any, Dict

import numpy as np
from omegaconf import DictConfig, ListConfig
from pydantic import BaseModel, Field, validator

# Ignore import deprecation warnings from ray.
warnings.filterwarnings("ignore", category=DeprecationWarning)

from ray.rllib.agents.a3c import A2CTrainer  # noqa
from ray.rllib.agents.a3c import A3CTrainer  # noqa
from ray.rllib.agents.dqn import ApexTrainer, R2D2Trainer  # noqa
from ray.rllib.agents.impala import ImpalaTrainer  # noqa
from ray.rllib.agents.ppo import PPOTrainer  # noqa

from .environment import Environment  # noqa: E402


class Agent(BaseModel):
    """Represents the RL algorithm used."""

    # === Start of fields list. ===

    type: str = Field(allow_mutation=False)
    """The name of the class used to instantiate the RL algorithm as a string,
    e.g. :code:`"PPOTrainer". The class must be imported to this module to be
    used.
    """

    args: Dict[str, Any] = Field(default={}, allow_mutation=False)
    """A dictionary of arguments that are passed into the
    :code:`type` constructor.
    """

    checkpoint_freq: int = Field(default=1, ge=1, allow_mutation=False)
    """How frequently to checkpoint the agents progress, in rllib training
    iterations.
    """

    checkpoint_at_end: bool = Field(default=True, allow_mutation=False)
    """Whether to produce a final checkpoint at the end of training.
    """

    reuse_actors: bool = Field(default=True, allow_mutation=False)
    """Whether to reuse workers between training iterations."""

    # === Start of public API. ===

    @property
    def actual_type(self):
        """Get the trainer class type."""
        return self._to_class(self.type)

    @property
    def rllib_trainer_config_dict(self):
        """Merge generated arguments with user trainer args dict."""
        config = {
            "log_level": "INFO",
        }
        config.update(self.args)
        return config

    def make_agent(self, environment: Environment):
        """Construct an agent object."""
        try:
            return self.actual_type(config=self.args, env=environment.rllib_id)
        except TypeError as e:
            raise TypeError(
                "Error constructing RLlib trainer class "
                f"{self.actual_type.__name__}: {e}"
            ) from e

    def trainable_parameters_count(self, agent):
        """Given an agent instance (created by :code:`make_agent()`), compute
        and return the number of trainable parameters.
        """
        framework = self.args.get("framework")
        model = agent.get_policy().model
        if framework == "torch":
            return np.sum([np.prod(var.shape) for var in model.trainable_variables()])
        elif framework == "tf":
            return np.sum(
                [np.prod(v.get_shape().as_list()) for v in model.trainable_variables()]
            )
        raise ValueError(f"Unknown framework: {framework}")

    # === Start of implementation details. ===

    @staticmethod
    def _to_class(value):
        try:
            return globals()[value]
        except KeyError as e:
            raise ValueError(
                f"Unknown RLlib trainer class: {value}.\n"
                "Make sure it is imported in rl/model/agent.py"
            ) from e

    @validator("type")
    def validate_type(cls, value):
        cls._to_class(value)
        return value

    @validator("args", pre=True)
    def validate_args(cls, value):
        def omegaconf_to_py(x):
            if isinstance(x, DictConfig):
                return {k: omegaconf_to_py(v) for k, v in x.items()}
            elif isinstance(x, ListConfig):
                return [omegaconf_to_py(v) for v in x]
            else:
                return x

        return omegaconf_to_py(value)

    class Config:
        validate_assignment = True
