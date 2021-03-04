# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Simple compiler gym tabular q learning example.
Usage python tabular_q.py

Using selected features from Autophase observation space, given a specific training
program as gym environment, find the best action sequence using online q learning.
"""

import random

from absl import app, flags

# from compiler_gym.envs import CompilerEnv
from compiler_gym.util.flags.benchmark_from_flags import benchmark_from_flags
from compiler_gym.util.flags.env_from_flags import env_from_flags

flags.DEFINE_list(
    "actions",
    ["-sroa", "-mem2reg", "-newgvn"],
    "A list of action names to explore from.",
)
flags.DEFINE_float("discount", 0.99, "The discount factor.")
flags.DEFINE_list(
    "features_indices",
    [15, 18, 24],
    "Indices of Alphaphase features that are used to construct a state",
)
flags.DEFINE_float("learning_rate", 0.1, "learning rate of the q-learning.")
flags.DEFINE_int("episodes", 1000000, "number of episodes used to learn.")
flags.DEFINE_float("epsilon", 0.2, "Epsilon rate of exploration. ")
flags.DEFINE_integer("episode_length", 5, "The number of steps in each episode.")
FLAGS = flags.FLAGS


def hash_state_action(autophase_feature, action):
    return tuple(
        autophase_feature[FLAGS.features_indices] + [FLAGS.actions.index(action)]
    )


def get_env():
    return env_from_flags(benchmark_from_flags())


def select_action(q_table, ob, epsilon=0.0):
    qs = [q_table.get(hash_state_action(ob, act), -1) for act in FLAGS.actions]
    if random.random() < epsilon:
        return random.choice(FLAGS.actions)
    return FLAGS.actions[qs.index(max(qs))]


def train(q_table):
    # TODO fill this function
    return


def main(argv):
    q_table = {}
    train(q_table)
    env = get_env()
    FLAGS.observation = "autophase"
    ob = env.reset()
    # Roll out one episode and report the resulting policy.
    action_seq, rewards = [], []
    for _ in range(FLAGS.episode_length):
        a = select_action(q_table, ob)
        action_seq.append(a)
        ob, reward, done, info = env.step(env.action_space.index(a))
        rewards.append(reward)
    print("Resulting sequence: ", ",".join(action_seq), f"total reward{sum(rewards)}")


if __name__ == "__main__":
    app.run(main)
