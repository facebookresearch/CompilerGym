# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import ray
import yaml
from fasteners import InterProcessLock
from pydantic import BaseModel, Field
from ray import tune

from compiler_gym.util.executor import Executor
from compiler_gym.util.shell_format import indent, plural
from compiler_gym.util.statistics import geometric_mean

from .agent import Agent
from .environment import Environment
from .inference_result import InferenceResult
from .testing import Testing
from .training import Training

logger = logging.getLogger(__name__)


class Model(BaseModel):
    """The composition of the full environment, agent, training / testing
    regime, and execution environment. Provides the API for training / testing.
    """

    # === Start of fields list. ===

    executor: Executor
    """The execution environment to use for training / testing jobs."""

    environment: Environment = Field(allow_mutation=False)
    """Description of the environment, which defines the particular optimization
    problem, the reward signal for training, and the representation of state
    that the agent receives.
    """

    agent: Agent = Field(allow_mutation=False)
    """The agent describes the RLlib training algorithm that is used."""

    training: Training = Field(allow_mutation=False)
    """Description of the training regime: the benchmarks to learn over, and how
    long to learn for.
    """

    testing: Testing = Field(allow_mutation=False)
    """The testing setup."""

    working_directory: Path = Field(allow_mutation=False)
    """The working directory where logs and other artifacts are written to."""

    experiment: str = Field(default="unnamed_experiment", allow_mutation=False)
    """A logical name for this experiment. This is used for naming RLlib
    trials.
    """

    num_replicas: int = Field(default=1, ge=1, allow_mutation=False)
    """The number of duplicate jobs to run. E.g. for training, this will train
    :code:`n` independent models in trials that share the same working
    directory.
    """

    job_id: int = Field(default=0, allow_mutation=0)
    """An optional numeric job ID."""

    seed: int = Field(default=0xCC, allow_mutation=False)
    """The numeric seed to use"""

    compiler_gym_version: str = Field(default="", allow_mutation=False)
    """The compiler_gym.__version__ string."""

    # === Start of public API. ===

    def train(self) -> None:
        """Run the training job for this model."""
        logger.info("Model:\n%s", indent(self.yaml(), 4))

        logger.info("Starting training job in %s", self.working_directory)
        # The working directory may already have been created by hydra, so we
        # will check for the training-model.json file below to see if this
        # directory has already been used for training.
        self.working_directory.mkdir(parents=True, exist_ok=True)

        # Dump the parsed config to file.
        model_dump_path = self.working_directory / "training-model.json"
        assert not model_dump_path.is_file(), (
            f"Refusing to overwrite file: {model_dump_path}. "
            "Is the working directory clean?"
        )
        with open(model_dump_path, "w") as f:
            print(json.dumps(json.loads(self.json()), indent=2), file=f)

        with self.executor.get_executor(
            logs_dir=self.working_directory / "slurm",
            # Provision an extra hour for RLlib overhead.
            timeout_hours=self.training.timeout_hours + 1,
        ) as executor:
            for i in range(self.num_replicas):
                executor.submit(train_job, model=self, seed=self.seed + i, replica_id=i)

    def test_checkpoints(
        self, metric: str = "evaluation/episode_reward_mean"
    ) -> Iterable[Path]:
        df = self.dataframe
        if not len(df):
            return

        for logsdir in set(df["logsdir"].values):
            sdf = df[(df["logsdir"] == logsdir) & df["checkpoint"]]
            if not len(sdf):
                continue

            sdf = sdf.reset_index()
            idx = sdf[metric].idxmax()
            best = sdf.iloc[idx]

            logger.info(
                "Selected checkpoint %s with %s %f",
                best["checkpoint_path"],
                metric,
                best[metric],
            )
            yield Path(best["checkpoint_path"])

    def test(self) -> None:
        """Run the testing job for this model."""
        # Gather all the jobs to run now. We will submit them all in a batch.
        jobs = []
        for checkpoint in self.test_checkpoints():
            assert checkpoint.is_file(), f"Checkpoint not found: {checkpoint}"

            # Go up two levels to the main directory
            test_dir = checkpoint.parent.parent
            assert (test_dir / "progress.csv").is_file()

            # Make sure there aren't any other test jobs. Don't block on error,
            # just fail.
            lock = InterProcessLock(test_dir / ".test-lock")
            lock.acquire(blocking=False)

            # Try not to have to launch a job.
            if (test_dir / "test-meta.json").is_file():
                with open(test_dir / "test-meta.json") as f:
                    meta = json.load(f)
                    if meta.get("checkpoint") == checkpoint.name:
                        logger.info(
                            "Already have test results for %s, nothing to do",
                            checkpoint.name,
                        )
                        continue

            jobs.append((lock, checkpoint, test_dir))

        # Submit all the jobs now.
        with self.executor.get_executor(
            logs_dir=self.working_directory / "slurm",
            # Provision an extra hour for RLlib overhead.
            timeout_hours=self.testing.timeout_hours + 1,
            # Single threaded evaluation loop.
            cpus=2,
        ) as executor:
            for lock, checkpoint, test_dir in jobs:
                # Let go of the test lock. This begins the section of code where
                # races can occur if someone else comes along and grabs the lock
                # before the test job can take it over.
                lock.release()
                executor.submit(
                    test_job, model=self, checkpoint=checkpoint, outputs_dir=test_dir
                )

    def yaml(self) -> str:
        """Serialize the model configuration to a YAML string."""
        # We can't directly dump the dict() representation because we need to
        # simplify the types first, so we go via JSON.
        simplified_data = json.loads(self.json())
        return yaml.dump(simplified_data)

    @property
    def dataframe(self) -> pd.DataFrame:
        if not (self.working_directory / "train").is_dir():
            return pd.DataFrame([])

        dfs = []
        for subdir in (self.working_directory / "train").iterdir():
            if not subdir.is_dir():
                continue

            df = self._trial_to_dataframe(subdir)
            if df is not None:
                dfs.append(df)
                df.to_csv(subdir / "progress-redux.csv")

        return pd.concat(dfs) if dfs else pd.DataFrame([])

    def _trial_to_dataframe(self, directory: Path) -> Optional[pd.DataFrame]:
        components = directory.name.split("-")
        if len(components) < 3:
            logger.warning(
                "Directory name does not match expected "
                "{experiment}-{config}-{replica} format: %s",
                directory,
            )
            return

        replica = components[-1]
        config = components[-2]
        experiment = "-".join(components[:-2])

        if not (directory / "progress.csv").is_file():
            logger.warning("File not found: %s", directory / "progress.csv")
            return

        try:
            df = pd.read_csv(directory / "progress.csv")
        except pd.errors.EmptyDataError:
            return None

        df.insert(0, "logsdir", str(directory))
        df.insert(
            0,
            "experiment_timestamp",
            " ".join(
                [
                    self.working_directory.parent.parent.name,
                    self.working_directory.parent.name,
                ]
            ),
        )
        df.insert(0, "trial_name", directory.name)
        df.insert(0, "replica", replica)
        df.insert(0, "config", config)
        df.insert(0, "experiment", experiment)

        df["checkpoint"] = [
            (directory / f"checkpoint_{i:06d}").is_dir()
            for i in df["training_iteration"]
        ]
        df["checkpoint_path"] = [
            str(directory / f"checkpoint_{i:06d}" / f"checkpoint-{i}")
            if (directory / f"checkpoint_{i:06d}").is_dir()
            else None
            for i in df["training_iteration"]
        ]

        df["evaluation/episode_reward_geomean"] = [
            geometric_mean(eval(x)) for x in df["evaluation/hist_stats/episode_reward"]
        ]

        df["episode_reward_geomean"] = [
            geometric_mean(eval(x)) for x in df["hist_stats/episode_reward"]
        ]

        df["complete"] = [
            min(d / self.training.episodes, 1) for d in df["episodes_total"]
        ]

        df["cpus"] = self.executor.cpus
        df["gpus"] = self.executor.gpus

        df = df.set_index(["experiment", "config", "replica", "training_iteration"])

        return df

    @property
    def test_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Get a dictionary of test dataframes, keyed by trial name."""
        results = {}

        if not (self.working_directory / "train").is_dir():
            return results

        for subdir in (self.working_directory / "train").iterdir():
            if not subdir.is_dir():
                continue

            if not (subdir / "test-results.json").is_file():
                continue
            if not (subdir / "test-meta.json").is_file():
                continue

            with open(subdir / "test-meta.json") as f:
                meta = json.load(f)

            df = pd.read_json(subdir / "test-results.json")
            df["test_checkpoint"] = meta["checkpoint"]
            df["test_timestamp"] = meta["timestamp"]
            results[subdir.name] = df

        return results

    @classmethod
    def from_logsdir(cls, working_directory: Path) -> List["Model"]:
        """Reconstruct models by recursively reading from logs directories."""

        def find_models(dir: Path) -> Iterable[Path]:
            """Attempt to locate models recursively from logs directories."""
            if (dir / "training-model.json").is_file():
                yield dir / "training-model.json"
                return

            for entry in dir.iterdir():
                if entry.is_dir():
                    yield from find_models(entry)

        models: List[Model] = []
        for model_file in find_models(working_directory):
            with open(model_file) as f:
                try:
                    model = json.load(f)
                    model["working_directory"] = model_file.parent
                    models.append(cls(**model))
                except json.decoder.JSONDecodeError as e:
                    logger.warning(
                        "Failed to parse JSON for model file %s: %s", model_file, e
                    )
                    continue
        return models

    # === Start of implementation details. ===

    def make_rllib_trainer_config(self, seed: int) -> Dict[str, Any]:
        """Coerce user preferences into a dictionary of arguments for RLlib
        trainer class.
        """
        with self.environment.make_env() as env:
            evaluation_num_episodes = len(
                list(self.training.validation.benchmark_uris_iterator(env))
            )
            logger.info(
                "Calculated the number of episodes per evaluation to be %d",
                evaluation_num_episodes,
            )
            if not evaluation_num_episodes:
                raise ValueError("#. of validation episodes is 0!")

        derived_args = {
            "env": self.environment.rllib_id,
            "seed": seed,
            "horizon": self.environment.max_episode_steps,
            # Reserve one CPU for the trainer, the rest for rollout workers.
            "num_workers": self.executor.cpus - 1,
            "num_cpus_per_worker": 1,
            "num_gpus": self.executor.gpus,
            # Set the number of evaluation episodes to the size of the
            # validation set.
            "evaluation_num_episodes": evaluation_num_episodes,
            # 1 checkpoint = 1 evaluation.
            "evaluation_interval": self.agent.checkpoint_freq,
            # Argument dictionary passed to make_env().
            "env_config": {"type": "training"},
            "evaluation_config": {
                "env_config": {"type": "validation"},
            },
        }
        # Merge with the user args. In case of conflict, the user's arg value
        # overrides the derived arg value.
        return merge(derived_args, self.agent.args)

    class Config:
        validate_assignment = True


def test_job(model: Model, checkpoint: Path, outputs_dir: Path) -> None:
    init_logging()

    # Make sure there aren't any other test jobs. Don't block on error,
    # just fail.
    InterProcessLock(outputs_dir / ".test-lock").acquire(blocking=False)

    logger.info(
        "Initializing ray with 2 cpus and %d GPUs",
        model.executor.gpus,
    )
    ray.init(
        num_cpus=2,
        num_gpus=model.executor.gpus,
        include_dashboard=False,
    )

    tune.register_env(
        model.environment.rllib_id, lambda _: model.environment.make_env()
    )
    agent = model.agent.make_agent(model.environment)

    logger.info(
        "Restoring %s agent with %s trainable params from %s",
        model.agent.type,
        f"{model.agent.trainable_parameters_count(agent):,}",
        checkpoint,
    )
    agent.restore(str(checkpoint))

    # Run inference on all of the test benchmarks.
    results: List[InferenceResult] = []

    with model.environment.make_env() as env:
        test_benchmarks = list(model.testing.benchmark_uris_iterator(env))
        for i, benchmark in enumerate(test_benchmarks, start=1):
            env.reset(benchmark=benchmark)
            result = InferenceResult.from_agent(env, agent)
            logger.info(
                "Test %s of %s: %s",
                f"{i:,d}",
                f"{len(test_benchmarks):,d}",
                result,
            )
            results.append(result)

    # Do this once the actual work has been done so that failed jobs
    # don't leave meta files lying around.
    with open(outputs_dir / "test-results.json", "w") as f:
        json.dump([r.dict() for r in results], f)
    with open(outputs_dir / "test-meta.json", "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "checkpoint": checkpoint.name,
            },
            f,
        )

    # Explicit call to ray shutdown here so that multiple consecutive
    # jobs can initialize ray with different resource requirements.
    ray.shutdown()


def train_job(model: Model, seed: int, replica_id: int) -> None:
    init_logging()

    logger.info(
        "Initializing ray with %d %s and %d %s",
        model.executor.cpus,
        plural(model.executor.cpus, "CPU", "CPUs"),
        model.executor.gpus,
        plural(model.executor.gpus, "GPU", "GPUs"),
    )
    ray.init(
        num_cpus=model.executor.cpus,
        num_gpus=model.executor.gpus,
        include_dashboard=True,
    )

    logger.info("Registered RLlib environment %s", model.environment.rllib_id)

    def make_env(env_config: Dict[str, Any]):
        """Construct a training or validation environment."""
        env = model.environment.make_env()
        if "type" not in env_config:
            raise KeyError(f"No type in dict: {env_config}")
        if env_config["type"] == "training":
            return model.training.wrap_env(env)
        elif env_config["type"] == "validation":
            return model.training.validation.wrap_env(env)
        raise ValueError(f"Unknown environment type: {env_config['type']}")

    tune.register_env(
        model.environment.rllib_id,
        make_env,
    )

    def trial_name_creator(trial):
        del trial  # Unused
        # NOTE(cummins): Only a single trial per isntance.
        return f"{model.experiment}-C{model.job_id}-R{replica_id}"

    def trial_dirname_creator(trial):
        del trial  # Unused
        return f"{model.experiment}-C{model.job_id}-R{replica_id}"

    rllib_opts = {
        "config": model.make_rllib_trainer_config(seed),
        "time_budget_s": model.training.timeout_hours * 3600,
        "stop": {
            "episodes_total": model.training.episodes,
        },
        "reuse_actors": model.agent.reuse_actors,
        "checkpoint_freq": model.agent.checkpoint_freq,
        "checkpoint_at_end": model.agent.checkpoint_at_end,
        # Write RLlib files to: "<working_directory>/train/<experiment_name>-<job_id>".
        "local_dir": str(model.working_directory),
        "name": "train",
    }
    logger.info("RLlib options:\n%s", json.dumps(rllib_opts, indent=2))
    tune.run(
        model.agent.actual_type,
        trial_name_creator=trial_name_creator,
        trial_dirname_creator=trial_dirname_creator,
        **rllib_opts,
    )

    # Explicit call to ray shutdown here so that multiple consecutive
    # jobs can initialize ray with different resource requirements.
    ray.shutdown()


def merge(a, b, path=None):
    "Update values in `a` with values from `b`. Supported nested dicts."
    if path is None:
        path = []
    for key in b:
        if key in a and isinstance(a[key], dict) and isinstance(b[key], dict):
            merge(a[key], b[key], path + [str(key)])
        else:
            a[key] = b[key]
    return a


def init_logging():
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
