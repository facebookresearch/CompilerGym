# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Sweep the inner loop size of CUDA loop nests."""
import logging
from itertools import product
from pathlib import Path
from typing import List, Optional

from typer import Typer

import compiler_gym
from compiler_gym.util.runfiles_path import create_user_logs_dir

logger = logging.getLogger(__name__)

app = Typer()


def wrapped_step(env, action):
    done = True
    while done:
        observation, reward, done, info = env.step(action)
        if done:
            logger.warning("Step failed: %s", info["error_details"])
            env.reset()

    return observation, reward, done, info


def flops_after_steps(env, num_steps):
    wrapped_step(env, [1] * (num_steps - 1))
    env.observation_space = "flops"
    observation, _, _, _ = wrapped_step(env, 1)
    env.observation_space = None
    return observation


def run_one_sweep(
    device: str,
    k: int,
    vectorize: int = 1,
    linear: bool = False,
    logdir: Optional[Path] = None,
):
    """Run a single sweep."""
    logdir = logdir or create_user_logs_dir("loop_tool_sweep")
    logfile = logdir / f"k{k}-v{vectorize}-{device}-{'linear' if linear else 'log'}.txt"
    print("Logging results to", logfile)
    print()
    print("Device", "K", "Inner", "Vec.", "FLOPS", sep="\t")
    with open(logfile, "w") as f:
        print("device", "k", "inner", "vectorize", "flops", sep=",", file=f)

    def log(k, inner, vectorize, flops):
        print(device.upper(), k, inner, vectorize, flops, sep="\t", flush=True)
        with open(logfile, "a") as f:
            print(device, k, inner, vectorize, flops, sep=",", file=f)

    actions = [3, 0, 1, 3, 0]
    k *= 1024  # raw number of elements

    with compiler_gym.make("loop_tool-v0") as env:
        env.reset(
            benchmark=env.datasets.benchmark(
                uri=f"benchmark://loop_tool-{device}-v0/{k}"
            ),
            action_space="simple",
        )
        if vectorize - 1:
            vs = [1] * (vectorize - 1)
            actions += vs + [0, 1, 0] + vs + [0, 2, 0]
        for a in actions:
            wrapped_step(env, a)

        if linear:
            for i in range(k // (vectorize * 1024)):
                step_count = 1022 if i == 0 else 1023
                flops = flops_after_steps(env, step_count)
                log(k, (i + 1) * 1024, vectorize, flops)
        else:  # linear=False (log)
            inner = 1
            step = 512
            wrapped_step(env, [1] * (step - 1))
            inner += step - 1
            while inner * vectorize <= k:
                flops = flops_after_steps(env, step)
                inner += step
                log(k, inner, vectorize, flops)
                step *= 2


@app.command()
def sweep(
    device: List[str] = ["cuda"],
    k: List[int] = [512, 1024, 2048, 4096, 8192],
    vectorize: List[int] = [1],
    linear: List[bool] = [False],
    logdir: Optional[Path] = None,
):
    logdir = logdir or create_user_logs_dir("loop_tool_sweep")
    for device_, k_, vectorize_, linear_ in product(device, k, vectorize, linear):
        run_one_sweep(
            device=device_, k=k_, vectorize=vectorize_, linear=linear_, logdir=logdir
        )


if __name__ == "__main__":
    app()
