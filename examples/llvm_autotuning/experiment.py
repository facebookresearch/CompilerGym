# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List

import gym
import pandas as pd
import yaml
from llvm_autotuning.autotuners import Autotuner
from llvm_autotuning.benchmarks import Benchmarks
from pydantic import BaseModel, Field

from compiler_gym import CompilerEnvStateWriter
from compiler_gym.util.executor import Executor

logger = logging.getLogger(__name__)


class Experiment(BaseModel):
    """The composition of a full autotuning experiment, comprising autotuner,
    executor, and programs to tune.
    """

    # === Start of fields list. ===

    executor: Executor
    """The execution environment to use for training / testing jobs."""

    autotuner: Autotuner

    benchmarks: Benchmarks
    """The set of benchmarks to test on."""

    working_directory: Path
    """The working directory where logs and other artifacts are written to."""

    experiment: str = "unnamed_experiment"
    """A logical name for this experiment. This is used for naming RLlib
    trials.
    """

    num_replicas: int = Field(default=1, ge=1)
    """The number of duplicate jobs to run. E.g. for training, this will train
    :code:`n` independent models in trials that share the same working
    directory.
    """

    seed: int = 0xCC
    """A numeric random seed."""

    # === Start of public API. ===

    def run(self) -> None:
        """Run the experiment."""

        # The working directory may already have been created by hydra, so we
        # will check for the config.json file below to see if this experiment
        # has already run.
        self.working_directory.mkdir(parents=True, exist_ok=True)

        # Dump the parsed config to file.
        assert not self.config_path.is_file(), (
            f"Refusing to overwrite file: {self.config_path}. "
            "Is the working directory clean?"
        )
        with open(self.config_path, "w") as f:
            print(json.dumps(json.loads(self.json()), indent=2), file=f)
        logger.info("Wrote %s", self.config_path)

        results_num = 0
        with self.executor.get_executor(
            logs_dir=self.working_directory / "logs"
        ) as executor:
            with gym.make("llvm-v0") as env:
                for replica_num in range(self.num_replicas):
                    for benchmark in self.benchmarks.benchmark_uris_iterator(env):
                        results_num += 1
                        results_path = (
                            self.working_directory / f"results-{results_num:03d}.csv"
                        )
                        errors_path = (
                            self.working_directory / f"errors-{results_num:03d}.json"
                        )
                        executor.submit(
                            _experiment_worker,
                            autotuner=self.autotuner,
                            benchmark=benchmark,
                            results_path=results_path,
                            errors_path=errors_path,
                            seed=self.seed + replica_num,
                        )

    def yaml(self) -> str:
        """Serialize the model configuration to a YAML string."""
        # We can't directly dump the dict() representation because we need to
        # simplify the types first, so we go via JSON.
        simplified_data = json.loads(self.json())
        return yaml.dump(simplified_data)

    @property
    def config_path(self) -> Path:
        return self.working_directory / "config.json"

    @property
    def results_paths(self) -> Iterable[Path]:
        """Return an iterator over results files."""
        for path in self.working_directory.iterdir():
            if path.is_file() and path.name.startswith("results-"):
                yield path

    @property
    def errors(self) -> Iterable[Dict[str, str]]:
        """Return an iterator over errors.

        An error is a dictionary with keys: "benchmark", "error_type", and
        "error_message".
        """
        for path in self.working_directory.iterdir():
            if path.is_file() and path.name.startswith("errors-"):
                with open(path, "r") as f:
                    yield json.load(f)

    @property
    def configuration_number(self) -> str:
        return self.working_directory.name.split("-")[-1]

    @property
    def timestamp(self) -> str:
        return f"{self.working_directory.parent.parent.name}/{self.working_directory.parent.name}"

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return the results as a dataframe."""
        dfs = []
        for path in self.results_paths:
            dfs.append(pd.read_csv(path))

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs)

    @classmethod
    def from_logsdir(cls, working_directory: Path) -> List["Experiment"]:
        """Reconstruct experiments by recursively reading from logs directories."""

        def find_config_dumps(dir: Path) -> Iterable[Path]:
            """Attempt to locate config files recursively in directories."""
            if (dir / "config.json").is_file():
                yield dir / "config.json"
                return

            for entry in dir.iterdir():
                if entry.is_dir():
                    yield from find_config_dumps(entry)

        experiments: List[Experiment] = []
        for config_path in find_config_dumps(working_directory):
            with open(config_path) as f:
                try:
                    config = json.load(f)
                    config["working_directory"] = config_path.parent
                    experiments.append(cls(**config))
                except json.decoder.JSONDecodeError as e:
                    logger.warning(
                        "Failed to parse JSON for model file %s: %s", config, e
                    )
                    continue
        return experiments

    # === Start of implementation details. ===

    class Config:
        validate_assignment = True


def _experiment_worker(
    autotuner: Autotuner,
    benchmark: str,
    results_path: Path,
    errors_path: Path,
    seed: int,
) -> None:
    try:
        with autotuner.optimization_target.make_env(benchmark) as env:
            env.seed(seed)
            env.action_space.seed(seed)
            state = autotuner(env, seed=seed)
    except Exception as e:  # pylint: disable=broad-except
        logger.warning("Autotuner failed on benchmark %s: %s", benchmark, e)
        with open(errors_path, "w") as f:
            json.dump(
                {
                    "benchmark": benchmark,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
                f,
            )
        return

    logger.info("State %s", state)
    with CompilerEnvStateWriter(open(results_path, "w")) as writer:
        writer.write_state(state, flush=True)
