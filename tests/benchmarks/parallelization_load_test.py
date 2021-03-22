# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""A load test for measuring parallelization scalability.

This benchmark runs random episodes with varying numbers of parallel threads and
processes and records the time taken for each. The objective is to compare
performance of a simple random search when parallelized using thread-level
parallelism vs process-based parallelism.

This load test aims to provide a worst-case scenario for multithreading
performance testing: there is no communication or synchronization between
threads and the benchmark is entirely compute bound.
"""
from multiprocessing import Process, cpu_count
from threading import Thread

from absl import app, flags

from compiler_gym.util.flags.benchmark_from_flags import benchmark_from_flags
from compiler_gym.util.flags.env_from_flags import env_from_flags
from compiler_gym.util.timer import Timer

flags.DEFINE_integer("max_nproc", 2 * cpu_count(), "The maximum number of threads.")
flags.DEFINE_integer(
    "nproc_increment",
    cpu_count() // 4,
    "The number of workers to change at each step of the load test.",
)
flags.DEFINE_integer(
    "num_episodes", 50, "The number of episodes to run in each worker."
)
flags.DEFINE_integer("num_steps", 50, "The number of steps in each episode.")
flags.DEFINE_string(
    "logfile",
    "parallelization_load_test.csv",
    "The path of the file to write results to.",
)
FLAGS = flags.FLAGS


def run_random_search(num_episodes, num_steps) -> None:
    """The inner loop of a load test benchmark."""
    env = env_from_flags(benchmark=benchmark_from_flags())
    try:
        for _ in range(num_episodes):
            env.reset()
            for _ in range(num_steps):
                _, _, done, _ = env.step(env.action_space.sample())
                if done:
                    break
    finally:
        env.close()


def main(argv):
    assert len(argv) == 1, f"Unknown arguments: {argv[1:]}"

    with open(FLAGS.logfile, "w") as f:
        print(
            "nproc",
            "episodes_per_worker",
            "steps_per_episode",
            "total_episodes",
            "thread_steps_per_second",
            "process_steps_per_second",
            "thread_walltime",
            "process_walltime",
            sep=",",
            file=f,
        )

        for nproc in [1] + list(
            range(FLAGS.nproc_increment, FLAGS.max_nproc + 1, FLAGS.nproc_increment)
        ):
            # Perform the same `nproc * num_episodes` random trajectories first
            # using threads, then using processes.
            threads = [
                Thread(
                    target=run_random_search,
                    args=(FLAGS.num_episodes, FLAGS.num_steps),
                )
                for _ in range(nproc)
            ]
            with Timer(f"Run {nproc} threaded workers") as thread_time:
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

            processes = [
                Process(
                    target=run_random_search,
                    args=(FLAGS.num_episodes, FLAGS.num_steps),
                )
                for _ in range(nproc)
            ]
            with Timer(f"Run {nproc} process workers") as process_time:
                for process in processes:
                    process.start()
                for process in processes:
                    process.join()

            print(
                nproc,
                FLAGS.num_episodes,
                FLAGS.num_steps,
                FLAGS.num_episodes * nproc,
                (FLAGS.num_episodes * FLAGS.num_steps * nproc) / thread_time.time,
                (FLAGS.num_episodes * FLAGS.num_steps * nproc) / process_time.time,
                thread_time.time,
                process_time.time,
                sep=",",
                file=f,
                flush=True,
            )


if __name__ == "__main__":
    app.run(main)
