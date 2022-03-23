# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Simple parallelized random search."""
import json
import os
from multiprocessing import cpu_count
from pathlib import Path
from threading import Thread
from time import sleep, time
from typing import Callable, List, NamedTuple, Optional, Union

import humanize

from compiler_gym.envs import CompilerEnv
from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.service.connection import ServiceError
from compiler_gym.util import logs
from compiler_gym.util.gym_type_hints import ActionType
from compiler_gym.util.logs import create_logging_dir
from compiler_gym.util.tabulate import tabulate


class RandomSearchProgressLogEntry(NamedTuple):
    """A snapshot of incremental search progress."""

    runtime_seconds: float
    total_episode_count: int
    total_step_count: int
    num_passes: int
    reward: float

    def to_csv(self) -> str:
        return ",".join(
            [
                f"{self.runtime_seconds:.3f}",
                str(self.total_episode_count),
                str(self.total_step_count),
                str(self.num_passes),
                str(self.reward),
            ]
        )

    @classmethod
    def from_csv(cls, line: str) -> "RandomSearchProgressLogEntry":
        (
            runtime_seconds,
            total_episode_count,
            total_step_count,
            num_passes,
            reward,
        ) = line.split(",")
        return RandomSearchProgressLogEntry(
            float(runtime_seconds),
            int(total_episode_count),
            int(total_step_count),
            int(num_passes),
            float(reward),
        )


class RandomAgentWorker(Thread):
    """Worker thread to run a repeating agent.

    To stop the agent, set the :code:`alive` attribute of this thread to False.
    """

    def __init__(
        self,
        make_env: Callable[[], CompilerEnv],
        patience: int,
    ):
        super().__init__()
        self._make_env = make_env
        self._patience = patience

        # Incremental progress.
        self.total_environment_count = 0
        self.total_episode_count = 0
        self.total_step_count = 0
        self.best_returns = -float("inf")
        self.best_actions: List[ActionType] = []
        self.best_commandline: str = []
        self.best_found_at_time = time()

        self.alive = True  # Set this to False to signal the thread to stop.

    @property
    def should_run_one_episode(self) -> bool:
        """Whether to run an episode."""
        return self.alive or not self.total_episode_count

    def run(self) -> None:
        """Run episodes in an infinite loop."""
        while self.should_run_one_episode:
            self.total_environment_count += 1
            with self._make_env() as env:
                self._patience = self._patience or env.action_space.n
                self.run_one_environment(env)

    def run_one_environment(self, env: CompilerEnv) -> None:
        """Run random walks in an infinite loop. Returns if the environment ends."""
        while self.should_run_one_episode:
            self.total_episode_count += 1
            if not self.run_one_episode(env):
                return

    def run_one_episode(self, env: CompilerEnv) -> bool:
        """Run a single random episode.

        :param env: An environment.
        :return: True if the episode ended gracefully, else False.
        """
        observation = env.reset()
        actions: List[ActionType] = []
        patience = self._patience
        total_returns = 0
        while patience >= 0:
            patience -= 1
            self.total_step_count += 1
            # === Your agent here! ===
            action = env.action_space.sample()
            # === End of agent. ===
            actions.append(action)
            observation, reward, done, _ = env.step(action)
            if done:
                return False
            total_returns += reward
            if total_returns > self.best_returns:
                patience = self._patience
                self.best_returns = total_returns
                self.best_actions = actions.copy()
                try:
                    self.best_commandline = env.commandline()
                except NotImplementedError:
                    self.best_commandline = ""
                self.best_found_at_time = time()

        return True


def random_search(
    make_env: Callable[[], CompilerEnv],
    outdir: Optional[Union[str, Path]] = None,
    total_runtime: Optional[float] = 600,
    patience: int = 0,
    nproc: int = cpu_count(),
    skip_done: bool = False,
) -> CompilerEnv:
    with make_env() as env:
        env.reset()
        if not isinstance(env.unwrapped, CompilerEnv):
            raise TypeError(
                f"random_search() requires CompilerEnv. Called with: {type(env).__name__}"
            )

        benchmark_uri = env.benchmark.uri
        if not outdir:
            outdir = create_logging_dir(
                os.path.normpath(f"random/{benchmark_uri.scheme}/{benchmark_uri.path}")
            )
        outdir = Path(outdir)

        if not env.reward_space:
            raise ValueError("A reward space must be specified for random search")
        reward_space_name = env.reward_space.name

        action_space_names = list(env.action_space.names)

        metadata_path = outdir / logs.METADATA_NAME
        progress_path = outdir / logs.PROGRESS_LOG_NAME
        best_actions_path = outdir / logs.BEST_ACTIONS_NAME
        best_commandline_path = outdir / logs.BEST_COMMANDLINE_NAME

        if skip_done and metadata_path.is_file():
            # TODO(cummins): Return best reward.
            return 0

        # Write a metadata file.
        metadata = {
            "env": env.spec.id if env.spec else "",
            "benchmark": str(benchmark_uri),
            "reward": reward_space_name,
            "patience": patience,
        }
        with open(str(metadata_path), "w") as f:
            json.dump(metadata, f, sort_keys=True, indent=2)

    workers = [RandomAgentWorker(make_env, patience) for _ in range(nproc)]
    for worker in workers:
        worker.start()

    best_actions = []
    best_commandline = ""
    started = time()
    last_best_returns = -float("inf")

    print(
        f"Started {len(workers)} worker threads for {benchmark_uri} "
        f"using reward {reward_space_name}."
    )
    print(f"Writing logs to {outdir}")

    end_time = time() + total_runtime if total_runtime else None
    if end_time:
        print(f"=== Running for {humanize.naturaldelta(total_runtime)} ===")
    else:
        print("=== WARNING: This will loop forever! Use C-c to terminate. ===")
    print()  # Blank line gets filled below when the cursor moves up one line.

    try:
        with open(str(progress_path), "w") as f:
            print(
                "runtime_seconds",
                "total_episode_count",
                "total_step_count",
                "num_passes",
                "reward",
                sep=",",
                file=f,
                flush=True,
            )
            while not end_time or time() < end_time:
                sleep(0.5)
                total_episode_count = sum(
                    worker.total_episode_count for worker in workers
                )
                total_step_count = sum(worker.total_step_count for worker in workers)
                total_environment_count = sum(
                    worker.total_environment_count for worker in workers
                )

                best_worker = max(workers, key=lambda worker: worker.best_returns)
                best_returns = best_worker.best_returns
                best_actions = best_worker.best_actions
                best_commandline = best_worker.best_commandline
                runtime = time() - started
                print(
                    "\r\033[1A"
                    "\033[K"
                    f"Runtime: {humanize.naturaldelta(runtime)}. "
                    f"Num steps: {humanize.intcomma(total_step_count)} "
                    f"({humanize.intcomma(int(total_step_count / runtime))} / sec). "
                    f"Num episodes: {humanize.intcomma(total_episode_count)} "
                    f"({humanize.intcomma(int(total_episode_count / runtime))} / sec). "
                    f"Num restarts: {humanize.intcomma(total_environment_count - nproc)}.\n"
                    "\033[K"
                    f"Best reward: {best_returns:.4f} "
                    f"({len(best_actions)} passes, "
                    f"found after {humanize.naturaldelta(best_worker.best_found_at_time - started)})",
                    end="",
                    flush=True,
                )

                # Log the incremental progress improvements.
                if best_returns > last_best_returns:
                    entry = RandomSearchProgressLogEntry(
                        runtime_seconds=runtime,
                        total_episode_count=total_episode_count,
                        total_step_count=total_step_count,
                        num_passes=len(best_actions),
                        reward=best_returns,
                    )
                    print(entry.to_csv(), file=f, flush=True)
                    last_best_returns = best_returns

    except KeyboardInterrupt:
        print("\nkeyboard interrupt", end="", flush=True)

    best_action_names = [action_space_names[a] for a in best_actions]
    with open(str(best_actions_path), "w") as f:
        f.write("\n".join(best_action_names))
        f.write("\n")
    with open(str(best_commandline_path), "w") as f:
        print(best_commandline, file=f)
    print("\n", flush=True)

    print("Ending worker threads ... ", end="", flush=True)
    for worker in workers:
        worker.alive = False
    for worker in workers:
        try:
            worker.join()
        except ServiceError:
            # Service error can be raised on abrupt service termination causing
            # RPC errors.
            pass
    print("done")

    print("Replaying actions from best solution found:")
    with make_env() as env:
        env.reset()
        replay_actions(env, best_action_names, outdir)

    return env


def replay_actions(env: CompilerEnv, action_names: List[str], outdir: Path):
    logs_path = outdir / logs.BEST_ACTIONS_PROGRESS_NAME
    start_time = time()

    if isinstance(env, LlvmEnv):
        env.write_bitcode(outdir / "unoptimized.bc")

    with open(str(logs_path), "w") as f:
        ep_reward = 0
        for i, action in enumerate(action_names, start=1):
            _, reward, done, _ = env.step(env.action_space.names.index(action))
            assert not done
            ep_reward += reward
            print(
                f"Step [{i:03d} / {len(action_names):03d}]: reward={reward:.4f}   \t"
                f"episode={ep_reward:.4f}   \taction={action}"
            )
            progress = RandomSearchProgressLogEntry(
                runtime_seconds=time() - start_time,
                total_episode_count=1,
                total_step_count=i,
                num_passes=i,
                reward=reward,
            )
            print(progress.to_csv(), action, file=f, sep=",")

    if isinstance(env, LlvmEnv):
        env.write_bitcode(outdir / "optimized.bc")
        print(
            tabulate(
                [
                    (
                        "IR instruction count",
                        env.observation["IrInstructionCountO0"],
                        env.observation["IrInstructionCountOz"],
                        env.observation["IrInstructionCount"],
                    ),
                    (
                        "Object .text size (bytes)",
                        env.observation["ObjectTextSizeO0"],
                        env.observation["ObjectTextSizeOz"],
                        env.observation["ObjectTextSizeBytes"],
                    ),
                ],
                headers=("", "-O0", "-Oz", "final"),
            )
        )


def replay_actions_from_logs(env: CompilerEnv, logdir: Path, benchmark=None) -> None:
    best_actions_path = logdir / logs.BEST_ACTIONS_NAME
    meta_path = logdir / logs.METADATA_NAME

    assert best_actions_path.is_file(), f"File not found: {best_actions_path}"
    assert meta_path.is_file(), f"File not found: {meta_path}"

    with open(meta_path, "rb") as f:
        meta = json.load(f)

    with open(best_actions_path) as f:
        actions = [ln.strip() for ln in f.readlines() if ln.strip()]

    benchmark = benchmark or meta["benchmark"]
    env.reward_space = meta["reward"]
    env.reset(benchmark=benchmark)
    replay_actions(env, actions, logdir)
