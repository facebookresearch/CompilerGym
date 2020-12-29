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
from compiler_gym.util.tabulate import tabulate


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
            progress = logs.ProgressLogEntry(
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
        actions = [l.strip() for l in f.readlines() if l.strip()]

    benchmark = benchmark or meta["benchmark"]
    env.reward_space = meta["reward"]
    env.reset(benchmark=benchmark)
    replay_actions(env, actions, logdir)
    env.close()
