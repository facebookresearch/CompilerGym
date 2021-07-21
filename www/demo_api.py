"""Demonstration of wrapping CompilerGym in a simple Flask app.

This exposes an API with five operations:

   1. describe() -> dict  (/api/v2/describe)

        Describe the CompilerGym interface. This generates a list of action
        names and their numeric values, a list of benchmark datasets and the
        benchmarks within them, and a list of reward spaces.

   2. start(reward, benchmark) -> session_id, state
        (/api/v2/start/<reward>/<benchmark>)

        Start a session. This would happen when the user navigates to the page
        in their web browser. One tab = one session. Takes a reward space name
        and a benchmark URI as inputs. Returns a numeric session ID (this
        probably isn't the right way of doing things but I don't know any better
        :-) ). Also returns a state, which is the set of things we want to
        visualize to represent the current environment state.

   3. step(session_id, action) -> state  (/api/v2/<session_id>/<action>)

        Run an action and produce a new state, replacing the old one.

   4. undo(session_id) -> state  (/api/v2/<session_id>/undo)

        Undo the previous action, returning the previous state.

   5. stop(session_id)  (/api/v2/stop/<session_id>)

        End a session. This would be when the user closes the tab / disconnects.

To run this script, install the python dependencies using:

    pip install flask compiler_gym pydantic

Then launch it by running, in this directory:

    FLASK_APP=demo_api.py flask run

Interact with the API through GET requests, such as using curl. A "describe"
endpoint provides details on teh available actions, benchmarks, and rewards.:

    $ curl -s localhost:5000/api/v2/describe | jq
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

To start a session, specify a reward space and a benchmark. Note that this
requires URL-encoding the benchmark name as it contains slashes. e.g. to start a
new session using reward IrInstructionCountOz and benchmark
"benchmark://cbench-v1/qsort":

    $ curl -s localhost:5000/api/v2/start/IrInstructionCountOz/benchmark%3A%2F%2Fcbench-v1%2Fqsort | jq
    {
        "session_id": 0,
        "state": {
            "autophase": {
                "ArgsPhi": 10,
                ...
                "twoSuccessor": 31
            },
            "commandline": "opt  input.bc -o output.bc",
            "done": false,
            "instcount": {
                "AShrCount": 0,
                "AddCount": 9,
                ...
                "ZExtCount": 15
            },
            "ir": "; ModuleID = '-'\nsource_filename = \"-\"\ntarget ...",
            "reward": 0
        }
    }

That "state" dict contains the things that we would want to visualize in the
GUI. Our session ID is 0, lets take a step in this session using action "10":

    $ curl -s localhost:5000/api/v2/step/0/10 | jq
    {
        "state": {
            "autophase": {
                "ArgsPhi": 2,
                ..,
                "twoSuccessor": 29
            },
            "commandline": "opt -simplifycfg input.bc -o output.bc",
            "done": false,
            "instcount": {
                "AShrCount": 0,
                ...
                "ZExtCount": 15
            },
            "ir": "; ModuleID = '-'\nsource_filename = \"-\"\ntarget ...",
            "reward": 0.06501547987616099
        }
    }

Notice that the state dict has changed. Some of the numbers in the "autophase"
and "instcount" feature dictionary have changed, there is a reward value, and
the commandline now includes the flag needed to run action "10" (which turned
out to be the "-simplifycfg" flag).

We could carry on taking steps, or just end the session:

    $ curl -s localhost:5000/api/v2/stop/0
"""
from itertools import islice
from typing import Dict, List, Tuple

from flask import Flask, jsonify
from flask_cors import CORS
from pydantic import BaseModel

import compiler_gym
from compiler_gym import CompilerEnv

app = Flask("compiler_gym")
CORS(app)


class StateToVisualize(BaseModel):
    """Encapsulates everything we want to visualize in the frontend. This
    will change from step to step.
    """

    # This summarizes the sequence of actions that the user has selected so far:
    commandline: str

    # If the compiler environment dies, crashes, or encounters some
    # unrecoverable error, this "done" flag is set. At this point the user
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
    reward: float


# A set of sessions that are in use, keyed by a numeric session ID. Each session
# is represented by a list of (environment, state) tuples, whether the
# environment is a CompilerGym environment and the state is a StateToVisualize.
# Initially, a session consists of a single (environment, state) tuple. When an
# action is taken, this generates a new (environment, state) tuple that is
# appended the session list. In this way, undoing an operation is as simple as
# popping the most recent (environment, state) tuple from the list.
sessions: Dict[int, List[Tuple[CompilerEnv, StateToVisualize]]] = {}


def compute_state(env: CompilerEnv, actions: List[int]) -> StateToVisualize:
    """Apply a list of actions and produce a new state to visualize."""
    # This is where we get the compiler environment to do its thing, and compute
    # for us all of the features that we would like to visualize.
    (ir, instcount, autophase), (reward,), done, _ = env.raw_step(
        actions=actions,
        observations=[
            env.observation.spaces["Ir"],
            env.observation.spaces["InstCountDict"],
            env.observation.spaces["AutophaseDict"],
        ],
        rewards=[env.reward_space],
    )
    return StateToVisualize(
        commandline=env.commandline(),
        done=done,
        ir=ir,
        instcount=instcount,
        autophase=autophase,
        reward=reward,
    )


@app.route("/api/v2/describe")
def describe():
    env = compiler_gym.make("llvm-v0")
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
                        (x[len(dataset.name) + 1 :] for x in dataset.benchmark_uris()),
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


@app.route("/api/v2/start/<reward>/<path:benchmark>")
def start(reward: str, benchmark: str):
    env = compiler_gym.make("llvm-v0", benchmark=benchmark)
    env.reward_space = reward
    env.reset()
    state = compute_state(env, [])
    session_id = len(sessions)
    sessions[session_id] = [(env, state)]
    return jsonify({"session_id": session_id, "state": state.dict()})


@app.route("/api/v2/stop/<session_id>")
def stop(session_id: int):
    session_id = int(session_id)

    for env, _ in sessions[session_id]:
        env.close()
    del sessions[session_id]

    return jsonify({"session_id": session_id})


@app.route("/api/v2/step/<session_id>/<action>")
def step(session_id: int, action: int):
    session_id = int(session_id)
    action = int(action)

    session = sessions[session_id]
    new_env = session[-1][0].fork()
    new_state = compute_state(new_env, [action])
    session.append((new_env, new_state))

    return jsonify({"state": new_state.dict()})


@app.route("/api/v2/undo/<session_id>")
def undo(session_id: int):
    session_id = int(session_id)

    session = sessions[session_id]
    env, _ = session.pop()
    env.close()
    _, old_state = session[-1]

    return jsonify({"state": old_state.dict()})
