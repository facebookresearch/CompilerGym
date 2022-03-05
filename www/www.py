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

            all_states: An optional string that if "1" means that a list of
                all states will be returned, one for each action. Else, only
                the state for the final action is returned.

        Example usage:

            $ curl 'localhost:5000/api/v4/step?benchmark=benchmark://cbench-v1/adpcm&reward=IrInstructionCountOz&actions=1,2,3'
            {
                "commandline": "opt - ...",
                "rewards": [0.003],
                "done": false,
                "ir": "...",
                "states": [
                    {
                        "instcount": {...},
                        "autophase": {...},
                        "reward": 0.003
                    },
                ]
            }
"""
import logging
import os
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

    instcount: Dict[str, int]
    autophase: Dict[str, int]
    # The reward signal measures how "good" the previous action was. Over time
    # the sequence of actions that produces the highest cumulative reward is the
    # best:
    reward: float


class StepRequest(BaseModel):
    """User arguments to /api/v4/step."""

    # The name of the benchmark.
    benchmark: str

    # The reward space to use.
    reward: str

    # A comma-separated list of actions to perform.
    actions: List[int]

    # Whether to return a state for every action, or only the final action. See
    # StepReply.states.
    all_states: bool

    @classmethod
    def from_request(cls):
        """Parse the arguments from Flask's request arguments."""

        def required_arg(name: str) -> str:
            value = request.args.get(name)
            if not value:
                raise ValueError(f"Missing requirement argument: {name}")
            return value

        actions_str: str = request.args.get("actions")
        actions: List[int] = (
            [int(x) for x in actions_str.split(",")] if actions_str else []
        )

        return cls(
            benchmark=required_arg("benchmark"),
            reward=required_arg("reward"),
            actions=actions,
            all_states=request.args.get("all_states", "0") == "1",
        )


class StepReply(BaseModel):
    """The data returned by a call to /api/v4/step."""

    # This summarizes the sequence of actions that the user has selected so far:
    commandline: str

    # If the compiler environment dies, crashes, or encounters some
    # unrecoverable error, this "done" flag is set. At this point the user
    # should start a new session.
    done: bool

    # The current LLVM-IR:
    ir: str

    # A list of states to visualize, ordered from first to last.
    states: List[StateToVisualize]


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


def _step(request: StepRequest) -> StepReply:
    """Run the actual step with parsed arguments."""
    states: List[StateToVisualize] = []

    with env_lock:
        env.reward_space = request.reward
        env.reset(benchmark=request.benchmark)

        # Replay all actions except the last one.
        if request.all_states:
            # Replay actions one at a time to receive incremental rewards. The
            # first item represents the state prior to any actions.
            (instcount, autophase), _, done, info = env.raw_step(
                actions=[],
                observation_spaces=[
                    env.observation.spaces["InstCountDict"],
                    env.observation.spaces["AutophaseDict"],
                ],
            )
            if done:
                raise ValueError(
                    f"Failed to compute initial state: {info['error_details']}"
                )
            states.append(
                StateToVisualize(
                    instcount=instcount,
                    autophase=autophase,
                    reward=0,
                )
            )
            for action in request.actions[:-1]:
                (instcount, autophase), reward, done, info = env.step(
                    action,
                    observation_spaces=[
                        env.observation.spaces["InstCountDict"],
                        env.observation.spaces["AutophaseDict"],
                    ],
                )
                states.append(
                    StateToVisualize(
                        instcount=instcount,
                        autophase=autophase,
                        reward=reward,
                    )
                )
                if done:
                    raise ValueError(
                        f"Failed to apply action {action}: {info['error_details']}"
                    )
        else:
            # Replay actions in a single batch.
            _, _, done, info = env.step(request.actions[:-1])
            if done:
                raise ValueError(
                    f"Failed to apply actions {request.actions}: {info['error_details']}"
                )

        # Perform the final action.
        (ir, instcount, autophase), (reward,), done, _ = env.raw_step(
            actions=request.actions[-1:],
            observation_spaces=[
                env.observation.spaces["Ir"],
                env.observation.spaces["InstCountDict"],
                env.observation.spaces["AutophaseDict"],
            ],
            reward_spaces=[env.reward_space],
        )

    states.append(
        StateToVisualize(
            instcount=instcount,
            autophase=autophase,
            reward=reward,
        )
    )
    return StepReply(
        commandline=env.commandline(),
        done=done,
        ir=truncate(ir, max_line_len=250, max_lines=1024),
        states=states,
    )


@app.route("/api/v4/step")
def step() -> Dict[str, Any]:
    try:
        request = StepRequest.from_request()
    except ValueError as e:
        return jsonify({"error": f"Invalid actions: {e}"}), 400

    try:
        return jsonify(_step(request).dict())
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
