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

from compiler_gym.util.flags.benchmark_from_flags import benchmark_from_flags
from compiler_gym.util.flags.env_from_flags import env_from_flags

# from compiler_gym.util.debug_util import set_debug_level

# set_debug_level(3)

flags.DEFINE_list(
    "actions",
    [
        "-simplifycfg",
        "-reg2mem",
        "-early-cse-memssa",
        "-gvn-hoist",
        "-gvn",
        "-instsimplify",
        "-instcombine",
        "-jump-threading",
        "-loop-extract",
        "-loop-reduce",
        "-loop-versioning",
        "-newgvn",
        "-mem2reg",
        "-sroa",
        "-structurizecfg",
    ],
    "A list of action names to explore from.",
)
flags.DEFINE_float("discount", 1.0, "The discount factor.")
flags.DEFINE_list(
    "features_indices",
    [19, 22, 51],
    "Indices of Alphaphase features that are used to construct a state",
)
flags.DEFINE_float("learning_rate", 0.1, "learning rate of the q-learning.")
flags.DEFINE_integer("episodes", 1000, "number of episodes used to learn.")
flags.DEFINE_float("epsilon", 0.2, "Epsilon rate of exploration. ")
flags.DEFINE_integer("episode_length", 5, "The number of steps in each episode.")
FLAGS = flags.FLAGS


def hash_state_action(autophase_feature, action):
    return tuple(
        (tuple(autophase_feature[FLAGS.features_indices]), FLAGS.actions.index(action))
    )


def get_env():
    return env_from_flags(benchmark_from_flags())


def select_action(q_table, ob, epsilon=0.0):
    qs = [q_table.get(hash_state_action(ob, act), -1) for act in FLAGS.actions]
    if random.random() < epsilon:
        return random.choice(FLAGS.actions)
    max_indices = [i for i, x in enumerate(qs) if x == max(qs)]
    # Breaking ties at random by selecting any of the indices.
    return FLAGS.actions[random.choice(max_indices)]


def get_max_q_value(q_table, ob):
    max_q = 0
    for act in FLAGS.actions:
        hashed = hash_state_action(ob, act)
        if hashed not in q_table:
            q_table[hashed] = 0
        max_q = max(q_table[hashed], max_q)
    return max_q


def train(q_table):
    # Buffer an old version of q table to inspect training progress
    prev_q = {}

    for i in range(FLAGS.episodes):
        current_length = 0
        env = get_env()
        obs = env.reset()
        while current_length < FLAGS.episode_length:
            # Run Epsilon greedy policy.
            a = select_action(q_table, obs, FLAGS.epsilon)
            hashed = hash_state_action(obs, a)
            if hashed not in q_table:
                q_table[hashed] = 0
            obs, reward, done, info = env.step(env.action_space.flags.index(a))
            # print({i:obs[i]-obs_prev[i] for i in range(obs.shape[0])}, reward, a)
            current_length += 1
            # Get max q at new state.
            target = reward + FLAGS.discount * get_max_q_value(q_table, obs)
            # Update Q value at current state action pair.
            q_table[hashed] = (
                FLAGS.learning_rate * target
                + (1 - FLAGS.learning_rate) * q_table[hashed]
            )

        if i % 50 == 0:
            print(f"Running episode {i}, current Q table: ", q_table)

            def compare_qs(q_old, q_new):
                diff = [q_new[k] - v for k, v in q_old.items()]
                return sum(diff) / len(diff) if diff else "NaN"

            print(
                f"Newly added Q entries {len(q_table)-len(prev_q)}, averaged diff {compare_qs(prev_q, q_table)}"
            )
            if compare_qs(prev_q, q_table) < 0.1:
                break
            prev_q = q_table.copy()


def setup_env():
    FLAGS.observation = "Autophase"
    FLAGS.reward = "IrInstructionCount"
    FLAGS.benchmark = "cBench-v0/dijkstra"
    FLAGS.env = "llvm-v0"


def main(argv):
    setup_env()
    # Train a Q table.
    q_table = {}
    try:
        train(q_table)
        # Rollout based on the Max-Q policy.
        env = get_env()
        ob = env.reset()
        # Roll out one episode and report the resulting policy.
        action_seq, rewards = [], []
        for _ in range(FLAGS.episode_length):
            a = select_action(q_table, ob)
            action_seq.append(a)
            ob, reward, done, info = env.step(env.action_space.flags.index(a))

            rewards.append(reward)
        print(
            "Resulting sequence: ", ",".join(action_seq), f"total reward{sum(rewards)}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    app.run(main)
