# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Run a parallelized brute force of an action space.

This script enumerates all possible combinations of actions up to a finite
length and evaluates them, logging the incremental rewards of each.

Example usage:

    $ python brute_force.py --env=llvm-ic-v0 --benchmark=cbench-v1/dijkstra \
          --episode_length=8 --brute_force_action_list=-sroa,-mem2reg,-newgvn

    Enumerating all episodes of 3 actions x 8 steps
    Started 24 brute force workers for benchmark benchmark://cbench-v1/dijkstra using reward IrInstructionCountOz.
    === Running 6,561 trials ===
    Runtime: 8 seconds. Progress: 100.00%. Best reward found: 0.8571428571428572.
    Ending jobs ... I1014 12:04:51.671775 3245811 CreateAndRunCompilerGymServiceImpl.h:128] Service "/dev/shm/compiler_gym_cec/s/1014T120451-646797-5770" listening on 37505, PID = 3245811
    completed 6,561 of 6,561 trials (100.000%), best sequence -mem2reg -mem2reg -sroa -sroa -mem2reg -sroa -sroa -newgvn

Use --help to list the configurable options.
"""
import itertools
import json
import logging
import math
import os
import sys
from pathlib import Path
from queue import Queue
from threading import Thread
from time import time
from typing import List

import humanize
from absl import app, flags

import compiler_gym.util.flags.episode_length  # noqa Flag definition.
import compiler_gym.util.flags.nproc  # noqa Flag definition.
import compiler_gym.util.flags.output_dir  # noqa Flag definition.
from compiler_gym.envs import CompilerEnv
from compiler_gym.util.flags.benchmark_from_flags import benchmark_from_flags
from compiler_gym.util.flags.env_from_flags import env_from_flags
from compiler_gym.util.logs import create_logging_dir

flags.DEFINE_list(
    "brute_force_action_list",
    [],
    "A list of action names to enumerate. If not provided, all actions are used "
    "(warning: this might make a long time!)",
)

FLAGS = flags.FLAGS


def grouper(iterable, n):
    """Split an iterable into chunks of length `n`, padded if required."""
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=None)


class BruteForceProducer(Thread):
    """A thread which enumerates all possible combinations of actions up to
    length episode_length and writes chunks of these combinations to a queue.
    """

    def __init__(
        self,
        in_q: Queue,
        actions: List[int],
        episode_length: int,
        nproc: int,
        chunksize: int = 128,
    ):
        super().__init__()
        self.in_q = in_q
        self.actions = actions
        self.episode_length = episode_length
        self.nproc = nproc
        self.chunksize = chunksize

        self.alive = True  # Set this to False to signal the thread to stop.

    def run(self):
        for chunk in grouper(
            itertools.product(*[self.actions] * self.episode_length), self.chunksize
        ):
            if not self.alive:
                break
            self.in_q.put(chunk)

        # Signal for each worker to end.
        for _ in range(self.nproc):
            self.in_q.put(None)


class BruteForceWorker(Thread):
    """Worker thread which reads chunks of action lists and evaluates them.

    Chunks of action lists are read from in_q and written to out_q, along with
    the incremental reward of each action.
    """

    def __init__(
        self,
        worker_id: int,
        in_q: Queue,
        out_q: Queue,
        env: CompilerEnv,
    ):
        super().__init__()
        self.id = worker_id
        self.in_q = in_q
        self.out_q = out_q
        self.env = env

        # Incremental progress.
        self.num_trials = 0

        self.alive = True  # Set this to False to signal the thread to stop.

    def log(self, *args, **kwargs):
        logging.debug(
            f"Worker {self.id} ({self.num_trials} trials):", *args, **kwargs, flush=True
        )

    def run(self) -> None:
        """Grab chunks of work from in_q and write results to out_q."""
        chunk = self.in_q.get()
        while chunk and self.alive:
            results = []
            self.log("Processing chunk")
            for actions in chunk:
                # A "None" value is used to pad an incomplete chunk. There will
                # be no more work to do after this.
                if not actions:
                    break
                self.num_trials += 1
                rewards = self.run_one_episode(actions)
                results.append((actions, rewards))
            self.out_q.put(results)
            chunk = self.in_q.get()

        # Signal that we're done.
        self.out_q.put(None)
        self.env.close()
        self.log("Worker is done")

    def run_one_episode(self, actions: List[int]) -> List[float]:
        """Evaluate the reward of every action in a list."""
        self.env.reset()
        rewards = []
        for action in actions:
            _, reward, done, _ = self.env.step(action)
            rewards.append(reward)
            if done:
                break
        return rewards


def run_brute_force(
    make_env,
    action_names: List[str],
    episode_length: int,
    outdir: Path,
    nproc: int,
    chunksize: int = 128,
):
    """Run a brute force job."""
    meta_path = outdir / "meta.json"
    results_path = outdir / "results.csv"

    with make_env() as env:
        env.reset()

        action_names = action_names or env.action_space.names

        if not env.reward_space:
            raise ValueError("A reward space must be specified for random search")
        reward_space_name = env.reward_space.id

        actions = [env.action_space.names.index(a) for a in action_names]
        benchmark_uri = str(env.benchmark)

        meta = {
            "env": env.spec.id,
            "action_names": action_names,
            "benchmark": benchmark_uri,
            "reward": reward_space_name,
            "init_reward": env.reward[reward_space_name],
            "episode_length": episode_length,
            "nproc": nproc,
            "chunksize": chunksize,
        }
        with open(str(meta_path), "w") as f:
            json.dump(meta, f)
        print(f"Wrote {meta_path}")
        print(f"Writing results to {results_path}")

    # A queue for communicating action sequences to workers, and a queue for
    # workers to report <action_sequence, reward_sequence> results.
    in_q = Queue(maxsize=32)
    out_q = Queue(maxsize=128)

    # Generate the action sequences to run.
    producer = BruteForceProducer(
        in_q=in_q,
        nproc=nproc,
        actions=actions,
        episode_length=episode_length,
        chunksize=chunksize,
    )
    producer.start()

    # Worker threads that will consume the action sequences and produce rewards.
    workers = [
        BruteForceWorker(worker_id=i, env=make_env(), in_q=in_q, out_q=out_q)
        for i in range(1, nproc + 1)
    ]
    for worker in workers:
        worker.start()

    # The consumer loop. Read results from workers as they come in and write
    # them to file.
    started = time()
    expected_trial_count = len(actions) ** episode_length
    expected_chunk_count = math.ceil(expected_trial_count / chunksize)
    chunk_count = 0
    best_reward = -float("inf")
    best_action_sequence = []
    print(
        f"Enumerating all episodes of {len(actions)} actions x {episode_length} steps"
    )
    print(
        f"Started {len(workers)} brute force workers for benchmark "
        f"{benchmark_uri} using reward {reward_space_name}."
    )
    print(f"=== Running {humanize.intcomma(expected_trial_count)} trials ===")
    try:
        with open(str(results_path), "w") as f:
            print(
                *[f"action_{i}" for i in range(1, episode_length + 1)],
                *[f"reward_{i}" for i in range(1, episode_length + 1)],
                sep=",",
                file=f,
                flush=True,
            )

            nproc_completed = 0
            while nproc_completed < nproc:
                chunk = out_q.get()
                if not chunk:
                    nproc_completed += 1
                    continue
                chunk_count += 1
                print(
                    f"\r\033[KRuntime: {humanize.naturaldelta(time() - started)}. "
                    f"Progress: {chunk_count/expected_chunk_count:.2%}. "
                    f"Best reward found: {best_reward}.",
                    file=sys.stderr,
                    flush=True,
                    end="",
                )
                for actions, rewards in chunk:
                    print(*actions, *rewards, sep=",", file=f, flush=True)
                    if rewards and rewards[-1] is not None:
                        if sum(rewards) > best_reward:
                            best_reward = sum(rewards)
                            best_action_sequence = actions
    except KeyboardInterrupt:
        print("\nkeyboard interrupt", end="", flush=True)

    print(file=sys.stderr, flush=True)
    print("Ending jobs ... ", end="", flush=True)

    # In case of early exit, signal to the threads to terminate.
    producer.alive = False
    for worker in workers:
        worker.alive = False

    # Wait for everyone to finish.
    producer.join()
    for worker in workers:
        worker.join()

    num_trials = sum(worker.num_trials for worker in workers)
    with make_env() as env:
        print(
            f"completed {humanize.intcomma(num_trials)} of "
            f"{humanize.intcomma(expected_trial_count)} trials "
            f"({num_trials / expected_trial_count:.3%}), best sequence",
            " ".join([env.action_space.flags[i] for i in best_action_sequence]),
        )


def main(argv):
    """Main entry point."""
    argv = FLAGS(argv)
    if len(argv) != 1:
        raise app.UsageError(f"Unknown command line arguments: {argv[1:]}")

    # Use default logdir of <base>/brute_force/<benchmark> unless told
    # otherwise.
    benchmark = benchmark_from_flags()
    if not benchmark:
        raise app.UsageError("No benchmark specified.")

    with env_from_flags(benchmark) as env:
        env.reset()
        logs_dir = Path(
            FLAGS.output_dir
            or create_logging_dir(
                f'brute_force/{os.path.normpath(f"random/{env.benchmark.uri.scheme}/{env.benchmark.uri.path}")}'
            )
        )

    run_brute_force(
        make_env=lambda: env_from_flags(benchmark_from_flags()),
        action_names=FLAGS.brute_force_action_list,
        episode_length=FLAGS.episode_length,
        outdir=logs_dir,
        nproc=FLAGS.nproc,
    )


if __name__ == "__main__":
    app.run(main)
