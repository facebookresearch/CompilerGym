# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This modules defines a class for describing LLVM autotuners."""
import tempfile
from pathlib import Path
from typing import Any, Dict

from llvm_autotuning.autotuners.greedy import greedy  # noqa autotuner
from llvm_autotuning.autotuners.nevergrad_ import nevergrad  # noqa autotuner
from llvm_autotuning.autotuners.opentuner_ import opentuner_ga  # noqa autotuner
from llvm_autotuning.autotuners.random_ import random  # noqa autotuner
from llvm_autotuning.optimization_target import OptimizationTarget
from pydantic import BaseModel, validator

from compiler_gym.compiler_env_state import CompilerEnvState
from compiler_gym.envs import CompilerEnv
from compiler_gym.util.capture_output import capture_output
from compiler_gym.util.runfiles_path import transient_cache_path
from compiler_gym.util.temporary_working_directory import temporary_working_directory
from compiler_gym.util.timer import Timer


class Autotuner(BaseModel):
    """This class represents an instance of an autotuning algorithm.

    After instantiating from a config dict, instances of this class can be used
    to tune CompilerEnv instances:

        >>> autotuner = Autotuner(
            algorithm="greedy",
            optimization_target="codesize",
            search_time_seconds=1800,
        )
        >>> env = compiler_gym.make("llvm-v0")
        >>> autotuner(env)
    """

    algorithm: str
    """The name of the autotuner algorithm."""

    optimization_target: OptimizationTarget
    """The target that the autotuner is optimizing for."""

    search_time_seconds: int
    """The search budget of the autotuner."""

    algorithm_config: Dict[str, Any] = {}
    """An optional dictionary of keyword arguments for the autotuner function."""

    @property
    def autotune(self):
        """Return the autotuner function for this algorithm.

        An autotuner function takes a single CompilerEnv argument and optional
        keyword configuration arguments (determined by algorithm_config) and
        tunes the environment, returning nothing.
        """
        try:
            return globals()[self.algorithm]
        except KeyError as e:
            raise ValueError(
                f"Unknown autotuner: {self.algorithm}.\n"
                f"Make sure the {self.algorithm}() function definition is available "
                "in the global namespace of {__file__}."
            ) from e

    @property
    def autotune_kwargs(self) -> Dict[str, Any]:
        """Get the keyword arguments dictionary for the autotuner."""
        kwargs = {
            "optimization_target": self.optimization_target,
            "search_time_seconds": self.search_time_seconds,
        }
        kwargs.update(self.algorithm_config)
        return kwargs

    def __call__(self, env: CompilerEnv, seed: int = 0xCC) -> CompilerEnvState:
        """Autotune the given environment.

        :param env: The environment to autotune.

        :param seed: The random seed for the autotuner.

        :returns: A CompilerEnvState tuple describing the autotuning result.
        """
        # Run the autotuner in a temporary working directory and capture the
        # stdout/stderr.
        with tempfile.TemporaryDirectory(
            dir=transient_cache_path("."), prefix="autotune-"
        ) as tmpdir:
            with temporary_working_directory(Path(tmpdir)):
                with capture_output():
                    with Timer() as timer:
                        self.autotune(env, seed=seed, **self.autotune_kwargs)

        return CompilerEnvState(
            benchmark=env.benchmark.uri,
            commandline=env.commandline(),
            walltime=timer.time,
            reward=self.optimization_target.final_reward(env),
        )

    # === Start of implementation details. ===

    @validator("algorithm_config", pre=True)
    def validate_algorithm_config(cls, value) -> Dict[str, Any]:
        return value or {}
