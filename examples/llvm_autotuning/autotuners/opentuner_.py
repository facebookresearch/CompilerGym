# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
from llvm_autotuning.optimization_target import OptimizationTarget

from compiler_gym.envs import CompilerEnv
from compiler_gym.envs.llvm import compute_observation
from compiler_gym.service.connection import ServiceError
from compiler_gym.third_party.llvm import opt_path
from compiler_gym.util.runfiles_path import transient_cache_path

# Ignore import deprecation warnings from opentuner.
warnings.filterwarnings("ignore", category=DeprecationWarning)

import opentuner as ot  # noqa: E402
from opentuner import (  # noqa: E402
    ConfigurationManipulator,
    MeasurementInterface,
    PermutationParameter,
    Result,
)
from opentuner.search.binaryga import BinaryGA  # noqa: E402
from opentuner.search.manipulator import BooleanParameter  # noqa: E402
from opentuner.tuningrunmain import TuningRunMain  # noqa: E402


def opentuner_ga(
    env: CompilerEnv,
    optimization_target: OptimizationTarget,
    search_time_seconds: int,
    seed: int,
    max_copies_of_pass: int = 4,
    population: int = 200,
    tournament: int = 5,
    mutate: int = 2,
    sharing: int = 1,
    **kwargs,
) -> None:
    """Optimize an environment using opentuner.

    OpenTuner is an extensible framework for program autotuning:

        https://opentuner.org/
    """
    cache_dir = transient_cache_path("llvm_autotuning")
    cache_dir.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=cache_dir, prefix="opentuner-") as tmpdir:
        argparser = ot.default_argparser()
        args = argparser.parse_args(
            args=[
                f"--stop-after={search_time_seconds}",
                f"--database={tmpdir}/opentuner.db",
                "--no-dups",
                "--technique=custom",
                f"--seed={seed}",
                "--parallelism=1",
            ]
        )
        ot.search.technique.register(
            BinaryGA(
                population=population,
                tournament=tournament,
                mutate=mutate,
                sharing=sharing,
                name="custom",
            )
        )
        manipulator = LlvmOptFlagsTuner(
            args,
            target=optimization_target,
            benchmark=env.benchmark,
            max_copies_of_pass=max_copies_of_pass,
        )
        tuner = TuningRunMain(manipulator, args)
        tuner.main()

        class DesiredResult:
            def __init__(self, configuration) -> None:
                self.configuration = configuration

        class Configuration:
            def __init__(self, data) -> None:
                self.data = data

        wrapped = DesiredResult(Configuration(manipulator.best_config))
        manipulator.run(wrapped, None, None)
        env.reset()
        env.step(manipulator.serialize_actions(manipulator.best_config))


class LlvmOptFlagsTuner(MeasurementInterface):
    def __init__(
        self,
        *args,
        target: OptimizationTarget,
        benchmark=None,
        max_copies_of_pass=4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.opt = str(opt_path())

        self.env = target.make_env(benchmark)
        self.env.reset()
        self.target = target
        self.observation_space = self.env.observation.spaces[
            target.optimization_space_enum_name
        ]

        self.unoptimized_path = str(
            self.env.service.connection.working_dir / "opentuner-unoptimized.bc"
        )
        self.tmp_optimized_path = str(
            self.env.service.connection.working_dir / "opentuner-optimized.bc"
        )
        self.env.write_bitcode(self.unoptimized_path)
        self.env.write_bitcode(self.tmp_optimized_path)

        self.cost_o0 = self.env.observation["IrInstructionCountO0"]
        self.cost_oz = self.env.observation["IrInstructionCountOz"]

        self.flags_limit = self.env.action_space.n * max_copies_of_pass
        self.run_count = 0
        self.best_config = None

    def manipulator(self) -> ConfigurationManipulator:
        """Define the search space."""
        manipulator = ConfigurationManipulator()
        # A permutation parameter to order the passes that are present.
        manipulator.add_parameter(
            PermutationParameter("flag_order", list(range(self.flags_limit)))
        )
        # Boolean parameters for whether each pass is present.
        for i in range(self.flags_limit):
            manipulator.add_parameter(BooleanParameter(f"flag{i}"))

        def biased_random():
            cfg = ConfigurationManipulator.random(manipulator)
            # duplicates in the search space, bias to `n / 2` enabled
            disabled = random.sample(
                range(self.flags_limit), k=self.flags_limit - self.env.action_space.n
            )
            cfg.update({f"flag{x}": False for x in disabled})
            return cfg

        manipulator.random = biased_random

        return manipulator

    def serialize_flags(self, config):
        """Convert a point in the search space to an ordered list of opt flags."""
        return [self.env.action_space.flags[a] for a in self.serialize_actions(config)]

    def serialize_actions(self, config):
        """Convert a point in the search space to an ordered list of opt flags."""
        n = len(self.env.action_space.flags)
        serialized = []
        for i in config["flag_order"]:
            if config[f"flag{i}"]:
                serialized.append(i % n)
        return serialized

    def __del__(self):
        self.env.close()

    def run(self, desired_result, input, limit):
        """Run a single config."""
        del input  # Unused
        del limit  # Unused

        self.run_count += 1

        try:
            # Run opt to produce an optimized bitcode file.
            cmd = [
                self.opt,
                self.unoptimized_path,
                "-o",
                self.tmp_optimized_path,
            ]
            cmd += self.serialize_flags(desired_result.configuration.data)
            subprocess.check_call(
                cmd, timeout=300, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            if not Path(self.tmp_optimized_path).is_file():
                return Result(time=float("inf"))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return Result(time=float("inf"))

        # We need to jump through a couple of hoops to optimize for runtime
        # using OpenTuner. Replace the environment benchmark with the current
        # optimized file. Use the same benchmark protocol buffer so that any
        # dynamic configuration is preserved.
        if self.target == OptimizationTarget.RUNTIME:
            try:
                new_benchmark = self.env.benchmark
                new_benchmark.proto.program.uri = f"file:///{self.tmp_optimized_path}"
                self.env.reset(benchmark=new_benchmark)
                return Result(time=float(np.median(self.env.observation.Runtime())))
            except (ServiceError, TimeoutError):
                return Result(time=float("inf"))

        try:
            return Result(
                time=float(
                    compute_observation(self.observation_space, self.tmp_optimized_path)
                )
            )
        except (ValueError, TimeoutError):
            return Result(time=float("inf"))

    def save_final_config(self, configuration):
        # Save parameter for later.
        self.best_config = configuration.data
