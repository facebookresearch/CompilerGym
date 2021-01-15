# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Run a CompilerGym environment with text interface controls.

.. code-block::

    $ python -m compiler_gym.bin.manual_env --env=<env> [--benchmark=<name>] [--observation=<space>] [--reward=<space>]

The benchmark to use can be specified using :code:`--benchmark=<name>`. If not
provided, you be presented with a list of benchmarks to choose from on launch.
Select :code:`random` to select a benchmark randomly.
"""
import sys
from typing import Optional

from absl import app, flags

import compiler_gym.util.flags.ls_benchmark  # Flag definition.
from compiler_gym.envs import CompilerEnv
from compiler_gym.util import user_input
from compiler_gym.util.flags.benchmark_from_flags import benchmark_from_flags
from compiler_gym.util.flags.env_from_flags import env_from_flags
from compiler_gym.util.shell_format import emph
from compiler_gym.util.timer import Timer

FLAGS = flags.FLAGS


def run_manual_env(env: CompilerEnv):
    """Run an environment manually.

    The manual environment allows the user to step through the environment,
    selection observations, rewards, and actions to run as they see fit. This is
    useful for debugging.

    :param env: The environment to run.
    """
    benchmark = None
    if not env.benchmark:
        # Allow the user to choose a benchmark, with the first choice being
        # to select randomly.
        benchmarks = sorted(env.benchmarks)
        if not benchmarks:
            print(
                "No benchmarks available see https://facebookresearch.github.io/CompilerGym/getting_started.html#installing-benchmarks"
            )
            print("Exiting...")
            env.close()
            return

        # Strip default benchmark:// protocol.
        for i, benchmark in enumerate(benchmarks):
            if benchmark.startswith("benchmark://"):
                benchmarks[i] = benchmark[len("benchmark://") :]

        benchmark_index = user_input.read_list_index(
            "Benchmark", ["random"] + benchmarks
        )
        if benchmark_index:
            benchmark = benchmarks[benchmark_index - 1]
        else:
            benchmark = None

    with Timer() as timer:
        eager_observation = env.reset(benchmark=benchmark)

    print(f"Reset {env.benchmark} environment in {timer}")
    if env.observation_space and eager_observation is not None:
        print(f"Observation: {env.observation_space.to_string(eager_observation)}")

    observation_names = sorted(env.observation.spaces.keys())
    reward_names = sorted(env.reward.spaces.keys())
    last_eager_reward: Optional[float] = None
    step_count = 1

    while True:
        print(
            f"\nStep {step_count}. Select: [{emph('a')}]ction "
            f"[{emph('o')}]bservation [{emph('r')}]eward "
            f"[{emph('c')}]ommandline [{emph('e')}]nd >>> ",
            end="",
            flush=True,
        )
        while True:
            c = user_input.read_char()
            if c == "a":
                print("action", flush=True)
                index = user_input.read_list_index(
                    "Actions", ["random"] + env.action_space.names
                )
                step_count += 1
                with Timer() as t:
                    if index == 0:
                        # User selected "random" action.
                        index = env.action_space.sample()
                    else:
                        # Offset to remove "random" action from index.
                        index -= 1
                    eager_observation, eager_reward, done, info = env.step(index)

                # Print the eager observation, if available.
                if env.observation_space and eager_observation is not None:
                    print(
                        f"Observation: {env.observation_space.to_string(eager_observation)}"
                    )

                # Print the eager reward and the diff, if available.
                if env.reward_space and eager_reward is not None:
                    reward_diff = ""
                    if last_eager_reward is not None and eager_reward is not None:
                        reward_diff = (
                            f" (change: {eager_reward - last_eager_reward:.6f})"
                        )
                    print(f"Reward: {eager_reward:.6f}{reward_diff}")
                    last_eager_reward = eager_reward

                print(
                    f"Action {env.action_space.names[index]} in {t}.",
                    " No effect." if info.get("action_had_no_effect") else "",
                    flush=True,
                )
                if done:
                    print("Episode ended by environment: ", info["error_details"])
                    env.close()
                    return
                break
            if c == "o":
                print("observation", flush=True)
                observation_name = user_input.read_list_value(
                    "Observable values", observation_names
                )
                with Timer() as timer:
                    value = env.observation[observation_name]
                print(env.observation.spaces[observation_name].to_string(value))
                print(f"Observation {observation_name} in {timer}")
                break
            elif c == "r":
                print("reward", flush=True)
                reward_name = user_input.read_list_value("Rewards", reward_names)
                with Timer(f"Reward {reward_name}"):
                    print(f"{env.reward[reward_name]:.6f}")
                break
            elif c == "c":
                print("commandline")
                print("$", env.commandline(), flush=True)
                break
            elif c == "e":
                print("end", flush=True)
                with Timer("Closed environment"):
                    env.close()
                print("Have a nice day!")
                return


def main(argv):
    """Main entry point."""
    argv = FLAGS(argv)
    if len(argv) != 1:
        raise app.UsageError(f"Unknown command line arguments: {argv[1:]}")

    if FLAGS.ls_benchmark:
        benchmark = benchmark_from_flags()
        env = env_from_flags(benchmark)
        print("\n".join(sorted(env.benchmarks)))
        env.close()
        return

    with Timer("Initialized environment"):
        benchmark = benchmark_from_flags()
        env = env_from_flags(benchmark)

    run_manual_env(env)


if __name__ == "__main__":
    main(sys.argv)
