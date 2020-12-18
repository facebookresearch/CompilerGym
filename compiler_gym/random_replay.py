# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Replay the sequence of actions that produced the best reward."""
import json
import os
import shutil
from pathlib import Path
from time import time
from typing import List

from compiler_gym.envs import CompilerEnv, LlvmEnv
from compiler_gym.util import logs


def replay_actions(env: CompilerEnv, action_names: List[str], outdir: Path):
    logs_path = outdir / logs.BEST_ACTIONS_PROGRESS_NAME
    start_time = time()
    init_reward = env.reward[env.eager_reward_space]

    print(f"Step [{0:03d} / {len(action_names):03d}]: reward={init_reward:.2%}")

    with open(str(logs_path), "w") as f:
        progress = logs.ProgressLogEntry(
            runtime_seconds=time() - start_time,
            total_episode_count=1,
            total_step_count=0,
            num_passes=0,
            reward=init_reward,
        )
        print(progress.to_csv(), "", file=f, sep="")

        previous_reward = init_reward
        for i, action in enumerate(action_names, start=1):
            _, reward, done, _ = env.step(env.action_space.names.index(action))
            assert not done
            print(
                f"Step [{i:03d} / {len(action_names):03d}]: reward={reward:.2%}, "
                f"change={reward-previous_reward:.2%}, action={action}"
            )
            progress = logs.ProgressLogEntry(
                runtime_seconds=time() - start_time,
                total_episode_count=1,
                total_step_count=i,
                num_passes=i,
                reward=reward,
            )
            print(progress.to_csv(), action, file=f, sep=",")
            previous_reward = reward

    if isinstance(env, LlvmEnv):
        bitcode_path = outdir / logs.OPTIMIZED_BITCODE
        # Write optimized bitcode to file.
        temppath = env.observation["BitcodeFile"]
        # Copy, don't rename, since rename will fail if the paths are on
        # different devices.
        shutil.copyfile(temppath, str(bitcode_path))
        os.remove(temppath)


def replay_actions_from_logs(env: CompilerEnv, logdir: Path, benchmark=None) -> None:
    best_actions_path = logdir / logs.BEST_ACTIONS_NAME
    meta_path = logdir / logs.METADATA_NAME

    assert best_actions_path.is_file(), f"File not found: {best_actions_path}"
    assert meta_path.is_file(), f"File not found: {meta_path}"

    with open(meta_path, "rb") as f:
        meta = json.load(f)

    with open(best_actions_path) as f:
        actions = [l.strip() for l in f.readlines() if l.strip()]

    benchmark = benchmark or meta["benchmark"]
    env.eager_reward_space = meta["reward"]
    env.reset(benchmark=benchmark)
    replay_actions(env, actions, logdir)
    env.close()
