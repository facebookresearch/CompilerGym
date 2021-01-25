# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Simple parallelized random search."""
import json
from multiprocessing import cpu_count
from pathlib import Path
from threading import Thread
from time import sleep, time
from typing import Callable, List, Optional, Tuple, Union

import humanize

from compiler_gym.envs import CompilerEnv
from compiler_gym.random_replay import replay_actions
from compiler_gym.util import logs
from compiler_gym.util.logs import create_logging_dir


class RandomAgentWorker(Thread):
    """Worker thread to run a repeating agent.

    To stop the agent, set the alive attribute of this thread to False.
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
        self.best_actions: List[int] = []
        self.best_commandline: List[int] = []
        self.best_found_at_time = time()

        self.alive = True  # Set this to False to signal the thread to stop.

    def run(self) -> None:
        """Run episodes in an infinite loop."""
        while self.alive:
            self.total_environment_count += 1
            env = self._make_env()
            self.run_one_environment(env)
            env.close()

    def run_one_environment(self, env: CompilerEnv) -> None:
        """Run random walks in an infinite loop. Returns if the environment ends."""
        try:
            while self.alive:
                self.total_episode_count += 1
                if not self.run_one_episode(env):
                    return
        finally:
            env.close()

    def run_one_episode(self, env: CompilerEnv) -> bool:
        """Run a single random episode.

        :param env: An environment.
        :return: True if the episode ended gracefully, else False.
        """
        observation = env.reset()
        actions: List[int] = []
        patience = self._patience
        total_returns = 0
        while patience >= 0:
            patience -= 1
            self.total_step_count += 1
            # === Your agent here! ===
            action_index = env.action_space.sample()
            # === End of agent. ===
            actions.append(action_index)
            observation, reward, done, _ = env.step(action_index)
            if done:
                return False
            total_returns += reward
            if total_returns > self.best_returns:
                patience = self._patience
                self.best_returns = total_returns
                self.best_actions = actions.copy()
                self.best_commandline = env.commandline()
                self.best_found_at_time = time()

        return True


# Start of boilerplate code to run multiple agents and log progress.


def random_search(
    make_env: Callable[[], CompilerEnv],
    outdir: Optional[Union[str, Path]] = None,
    total_runtime: Optional[float] = 600,
    patience: int = 0,
    nproc: int = cpu_count(),
    skip_done: bool = False,
) -> Tuple[float, List[int]]:
    env = make_env()
    env.reset()
    if not isinstance(env, CompilerEnv):
        raise TypeError(
            f"random_search() requires CompilerEnv. Called with: {type(env).__name__}"
        )

    patience = patience or env.action_space.n
    benchmark_name = env.benchmark
    if not outdir:
        sanitized_benchmark_name = "/".join(benchmark_name.split("/")[-2:])
        outdir = create_logging_dir(f"random/{sanitized_benchmark_name}")
    outdir = Path(outdir)

    if not env.reward_space:
        raise ValueError("Eager reward must be specified for random search")
    reward_space_name = env.reward_space.id

    action_space_names = list(env.action_space.names)
    num_instructions = int(env.observation["IrInstructionCount"])

    metadata_path = outdir / logs.METADATA_NAME
    progress_path = outdir / logs.PROGRESS_LOG_NAME
    best_actions_path = outdir / logs.BEST_ACTIONS_NAME
    best_commandline_path = outdir / logs.BEST_COMMANDLINE_NAME

    if skip_done and metadata_path.is_file():
        # TODO(cummins): Return best reward.
        return 0

    # Write a metadata file.
    metadata = {
        "env": env.spec.id,
        "benchmark": benchmark_name,
        "reward": reward_space_name,
        "patience": patience,
        "num_instructions": num_instructions,
    }
    with open(str(metadata_path), "w") as f:
        json.dump(metadata, f, sort_keys=True, indent=2)

    env.close()

    workers = [RandomAgentWorker(make_env, patience) for _ in range(nproc)]
    for worker in workers:
        worker.start()

    best_actions = []
    best_commandline = ""
    started = time()
    last_best_returns = -float("inf")

    print(
        f"Started {len(workers)} worker threads for "
        f"{benchmark_name} ({humanize.intcomma(metadata['num_instructions'])} instructions) "
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
                    entry = logs.ProgressLogEntry(
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
    print(f"\n", flush=True)

    print("Ending worker threads ... ", end="", flush=True)
    for worker in workers:
        worker.alive = False
    for worker in workers:
        worker.join()
    print("done")

    print("Replaying actions from best solution found:")
    env = make_env()
    env.reset()
    replay_actions(env, best_action_names, outdir)
    env.close()

    return best_returns, best_actions
