# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Perform a random walk of the action space of a CompilerGym environment.

This program launches a CompilerGym service and runs a random number of steps, at
each step selecting a random action, and recording the observations and rewards
to stdout.

Example usage:

    # Build the CompilerGym service binary that you want to use.
    $ bazel build -c opt //examples/example_compiler_gym_service

    # Run a random walk.
    $ bazel run -c opt //compiler_gym/bin:random_walk -- \
        --local_service_binary=$PWD/bazel-bin/examples/example_compiler_gym_service/service/service \
        --program=foo \
        --observation=features \
        --reward=codesize \
        --step_min=100 \
        --step_max=100
"""
import random
from typing import List

import humanize
from absl import app, flags

from compiler_gym.envs import CompilerEnv
from compiler_gym.util.flags.benchmark_from_flags import benchmark_from_flags
from compiler_gym.util.flags.env_from_flags import env_from_flags
from compiler_gym.util.shell_format import emph
from compiler_gym.util.timer import Timer

flags.DEFINE_integer(
    "step_min",
    12,
    "The minimum number of steps. Fewer steps may be performed if the "
    "environment ends the episode early.",
)
flags.DEFINE_integer("step_max", 256, "The maximum number of steps.")
FLAGS = flags.FLAGS


def run_random_walk(env: CompilerEnv, step_count: int) -> List[float]:
    """Perform a random walk of the action space.

    :param env: The environment to use.
    :param step_count: The number of steps to run. This value is an upper bound -
        fewer steps will be performed if any of the actions lead the
        environment to end the episode.
    :return: The list of observed rewards.
    """
    rewards = []

    step_num = 0
    with Timer() as episode_time:
        env.reset()
        for step_num in range(1, step_count + 1):
            action_index = env.action_space.sample()
            with Timer() as step_time:
                observation, reward, done, info = env.step(action_index)
            print(f"\n=== Step {humanize.intcomma(step_num)} ===")
            print(
                f"Action:       {env.action_space.names[action_index]} "
                f"(changed={not info.get('action_had_no_effect')})"
            )
            rewards.append(reward)
            print(f"Reward:       {reward}")
            if env._default_observation:
                print(f"Observation:\n{observation}")
            print(f"Step time:    {step_time}")
            if done:
                print("Episode ended by environment")
                break
        env.close()

    def reward_delta(reward):
        delta = rewards[0] / max(reward, 1e-9) - 1
        return emph(f"{'+' if delta >= 0 else ''}{delta:.2%}")

    print(
        f"\nCompleted {emph(humanize.intcomma(step_num))} steps in {episode_time} "
        f"({step_num / episode_time.time:.1f} steps / sec)."
    )
    print(f"Init reward:  {rewards[0]}")
    print(f"Final reward: {rewards[-1]} ({reward_delta(rewards[-1])})")
    print(
        f"Max reward:   {max(rewards)} ({reward_delta(max(rewards))} "
        f"at step {humanize.intcomma(rewards.index(max(rewards)) + 1)})"
    )
    return rewards


def main(argv):
    """Main entry point."""
    assert len(argv) == 1, f"Unrecognized flags: {argv[1:]}"

    benchmark = benchmark_from_flags()
    env = env_from_flags(benchmark)

    step_min = min(FLAGS.step_min, FLAGS.step_max)
    step_max = max(FLAGS.step_min, FLAGS.step_max)
    run_random_walk(env=env, step_count=random.randint(step_min, step_max))


if __name__ == "__main__":
    app.run(main)
