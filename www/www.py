# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""A CompilerGym API and web frontend.

This exposes an API with two operations:

   1. /api/v4/describe

        Describe the CompilerGym interface. This generates a list of action
        names and their numeric values, a list of benchmark datasets and the
        benchmarks within them, and a list of reward spaces.

        Example usage:

            $ curl localhost:5000/api/v4/describe
            {
                "actions": {
                    "-adce": 1,
                    ...
                    "-tailcallelim": 122
                },
                "benchmarks": {
                    "benchmark://anghabench-v1": [
                        "8cc/extr_buffer.c_buf_append",
                        ...
                        "8cc/extr_buffer.c_quote_cstring_len"
                    ],
                    "benchmark://blas-v0": [
                        ...
                    ],
                    "benchmark://cbench-v1": [
                        "adpcm",
                        ...
                        "jpeg-c"
                    ],
                    ...
                },
                "rewards": [
                    "IrInstructionCount",
                    ...
                    "ObjectTextSizeOz"
                ]
            }

   2. /ap/v4/step

        Compute the state from the given environment description. Query
        arguments:

            benchmark: The name of the benchmark.

            reward: The name of the reward signal to use.

            actions: An optional, command-separated list of actions to run.

            all_rewards: An optional string that if "1" means that a list of
                all rewards will be returned, one for each action. Else, only
                the reward for the final action is returned.

        Example usage:

            $ curl 'localhost:5000/api/v4/step?benchmark=benchmark://cbench-v1/adpcm&reward=IrInstructionCountOz&actions=1,2,3'
            {
                "commandline": "opt - ...",
                "rewards": [0.003],
                "done": false,
                "ir": "...",
                "instcount": {...},
                "autophase": {...},
            }
"""
import logging
import os
import re
import sys
from itertools import islice
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from pydantic import BaseModel

import compiler_gym
from compiler_gym.envs import LlvmEnv
from compiler_gym.util.truncate import truncate

app = Flask("compiler_gym")
CORS(app)


resource_dir: Path = (Path(__file__).parent / "frontends/compiler_gym/build").absolute()

logger = logging.getLogger(__name__)

# A single compiler environment that is used to serve all endpoints.
env: LlvmEnv = compiler_gym.make("llvm-v0")
env_lock = Lock()


class StateToVisualize(BaseModel):
    """Encapsulates the state to visualize in the frontend."""

    # This summarizes the sequence of actions that the user has selected so far:
    commandline: str

    # If the compiler environment dies, crashes, or encounters some
    # unrecoverable error, this "done" flag is set. At this point the user d
    # should start a new session.
    done: bool

    # Observations that we would like to visualize. This list will grow over
    # time to include graphs and 2-D matrices:
    ir: str
    instcount: Dict[str, int]
    autophase: Dict[str, int]

    # The reward signal measures how "good" the previous action was. Over time
    # the sequence of actions that produces the highest cumulative reward is the
    # best:
    rewards: List[float]


@app.route("/api/v4/describe")
def describe():
    with env_lock:
        env.reset()
        return jsonify(
            {
                # A mapping from dataset name to benchmark name. To generate a full
                # benchmark URI, join the two values with a '/'. E.g. given a benchmark
                # "qsort" in the dataset "benchmark://cbench-v1", the full URI is
                # "benchmark://cbench-v1/qsort".
                "benchmarks": {
                    dataset.name: list(
                        islice(
                            (
                                x[len(dataset.name) + 1 :]
                                for x in dataset.benchmark_uris()
                            ),
                            10,
                        )
                    )
                    for dataset in env.datasets
                },
                # A mapping from the name of an action to the numeric value. This
                # numeric value is what is passed as argument to the step() function.
                "actions": {k: v for v, k in enumerate(env.action_space.flags)},
                # A list of reward space names. You select the reward space to use
                # during start().
                "rewards": sorted(list(env.reward.spaces.keys())),
            }
        )


def _step(
    benchmark: str, reward: str, actions: List[int], reward_history: bool
) -> StateToVisualize:
    rewards = []

    with env_lock:
        env.reward_space = reward
        env.reset()

        # Replay all actions except the last one.
        if reward_history:
            # Replay actions one at a time to receive incremental rewards. The first
            # reward represents the state prior to any actions.
            rewards.append(0)
            for action in actions[:-1]:
                _, reward, done, info = env.step(action)
                rewards.append(reward)
                if done:
                    raise ValueError(
                        f"Failed to apply action {action}: {info['error_details']}"
                    )
        else:
            # Replay actions in a single batch.
            _, _, done, info = env.step(actions[:-1])
            if done:
                raise ValueError(
                    f"Failed to apply actions {actions}: {info['error_details']}"
                )

        # Perform the final action.
        (ir, instcount, autophase), (reward,), done, _ = env.raw_step(
            actions=actions[-1:],
            observations=[
                env.observation.spaces["Ir"],
                env.observation.spaces["InstCountDict"],
                env.observation.spaces["AutophaseDict"],
            ],
            rewards=[env.reward_space],
        )

    rewards.append(reward)
    return StateToVisualize(
        commandline=env.commandline(),
        done=done,
        ir=truncate(ir, max_line_len=250, max_lines=1024),
        instcount=instcount,
        autophase=autophase,
        rewards=rewards,
    )


@app.route("/api/v4/step")
def step() -> Dict[str, Any]:
    actions_str: str = request.args.get("actions")
    benchmark: str = request.args.get("benchmark")
    reward: str = request.args.get("reward")
    reward_history: bool = request.args.get("reward_history", "0") == "1"

    try:
        actions: List[int] = (
            [int(x) for x in actions_str.split(",")] if actions_str else []
        )
    except ValueError as e:
        return jsonify({"error": f"Invalid actions: {e}"}), 400

    try:
        return jsonify(_step(benchmark, reward, actions, reward_history).dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Web endpoints.


@app.route("/")
def index_resource():
    return send_file(resource_dir / "index.html")


@app.route("/<path>")
def root_resource(path: str):
    return send_file(resource_dir / path)


@app.route("/static/css/<path>")
def css_resource(path: str):
    return send_file(resource_dir / "static/css/" / path)


@app.route("/static/js/<path>")
def js_resource(path: str):
    return send_file(resource_dir / "static/js/" / path)


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("Serving from %s", resource_dir)
    app.run(port=int(os.environ.get("PORT", "5000")), host="0.0.0.0")
