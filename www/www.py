"""A CompilerGym API and web frontend.

This exposes an API with five operations:

   1. describe() -> dict  (/api/v3/describe)

        Describe the CompilerGym interface. This generates a list of action
        names and their numeric values, a list of benchmark datasets and the
        benchmarks within them, and a list of reward spaces.

   2. start(reward, actions, benchmark) -> session_id, state[]
        (/api/v3/start/<reward>/<actions>/<benchmark>)

        Start a session. This would happen when the user navigates to the page
        in their web browser. One tab = one session. Takes a reward space name,
        a list of actions, and a benchmark URI as inputs. If no actions are to
        be performed, use "-". Returns a numeric session ID (this probably isn't
        the right way of doing things but I don't know any better :-) ). Also
        returns a list of states, which is the set of things we want to
        visualize to represent the current environment state. There is an
        initial state, and then one state for each action.

   3. step(session_id, actions) -> state[]  (/api/v3/<session_id>/<actions>)

        Run a list of actions and produce a list of states, replacing the old
        ones.

   4. undo(session_id, n) -> state  (/api/v3/<session_id>/undo/<n>)

        Undo `n` previous actions, returning the previous state.

   5. stop(session_id)  (/api/v3/stop/<session_id>)

        End a session. This would be when the user closes the tab / disconnects.

To run this script, install the python dependencies using:

    pip install flask compiler_gym pydantic

Then launch it by running, in this directory:

    FLASK_APP=demo_api.py flask run

Interact with the API through GET requests, such as using curl. A "describe"
endpoint provides details on teh available actions, benchmarks, and rewards.:

    $ curl -s localhost:5000/api/v3/describe | jq
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

    $ curl -s localhost:5000/api/v3/start/IrInstructionCountOz/benchmark%3A%2F%2Fcbench-v1%2Fqsort | jq
    {
        "session_id": 0,
        "states": [
            {
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
        ]
    }

That "state" dict contains the things that we would want to visualize in the
GUI. Our session ID is 0, lets take a step in this session using action "10":

    $ curl -s localhost:5000/api/v3/step/0/10 | jq
    {
        "states": [
            {
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
        ]
    }

Notice that the state dict has changed. Some of the numbers in the "autophase"
and "instcount" feature dictionary have changed, there is a reward value, and
the commandline now includes the flag needed to run action "10" (which turned
out to be the "-simplifycfg" flag).

We could carry on taking steps, or just end the session:

    $ curl -s localhost:5000/api/v3/stop/0
"""
import logging
import os
import sys
from itertools import islice
from pathlib import Path
from threading import Lock, Thread
from time import sleep, time
from typing import Dict, List, Tuple

from flask import Flask, jsonify, send_file
from flask_cors import CORS
from pydantic import BaseModel

import compiler_gym
from compiler_gym import CompilerEnv
from compiler_gym.util.truncate import truncate

app = Flask("compiler_gym")
CORS(app)


resource_dir: Path = (Path(__file__).parent / "frontends/compiler_gym/build").absolute()

logger = logging.getLogger(__name__)


class StateToVisualize(BaseModel):
    """Encapsulates everything we want to visualize in the frontend. This
    will change from step to step.
    """

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
    reward: float


class Session(BaseModel):
    states: List[Tuple[CompilerEnv, StateToVisualize]]
    last_use: float  # As returned by time().

    def close(self):
        for env, _ in self.states:
            env.close()

    class Config:
        arbitrary_types_allowed = True


# A set of sessions that are in use, keyed by a numeric session ID. Each session
# is represented by a list of (environment, state) tuples, whether the
# environment is a CompilerGym environment and the state is a StateToVisualize.
# Initially, a session consists of a single (environment, state) tuple. When an
# action is taken, this generates a new (environment, state) tuple that is
# appended the session list. In this way, undoing an operation is as simple as
# popping the most recent (environment, state) tuple from the list.
sessions: Dict[int, Session] = {}
sessions_lock = Lock()


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
        ir=truncate(ir, max_line_len=250, max_lines=1024),
        instcount=instcount,
        autophase=autophase,
        reward=reward,
    )


@app.route("/api/v3/describe")
def describe():
    with compiler_gym.make("llvm-v0") as env:
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


@app.route("/api/v3/start/<reward>/<actions>/<path:benchmark>")
def start(reward: str, actions: str, benchmark: str):
    env = compiler_gym.make("llvm-v0", benchmark=benchmark)
    env.reward_space = reward
    env.reset()
    state = compute_state(env, [])
    with sessions_lock:
        session_id = len(sessions)
        session = Session(states=[(env, state)], last_use=time())
        sessions[session_id] = session

    # Accept an optional comma-separated list of actions to compute and return.
    if actions != "-":
        step(session_id, actions)

    return jsonify(
        {
            "session_id": session_id,
            "states": [state.dict() for _, state in session.states],
        }
    )


@app.route("/api/v3/stop/<session_id>")
def stop(session_id: int):
    session_id = int(session_id)

    session = sessions[session_id]
    session.close()
    with sessions_lock:
        del sessions[session_id]

    return jsonify({"session_id": session_id})


@app.route("/api/v3/step/<session_id>/<actions>")
def step(session_id: int, actions: str):
    session_id = int(session_id)

    state_dicts = []
    session = sessions[session_id]
    for action in [int(a) for a in actions.split(",")]:
        new_env = session.states[-1][0].fork()
        new_state = compute_state(new_env, [action])
        session.states.append((new_env, new_state))
        state_dicts.append(new_state.dict())

    session.last_use = time()
    return jsonify({"states": state_dicts})


@app.route("/api/v3/undo/<session_id>/<n>")
def undo(session_id: int, n: int):
    session_id = int(session_id)
    n = int(n)

    session = sessions[session_id]
    for _ in range(n):
        env, _ = session.states.pop()
        env.close()
    _, old_state = session[-1]

    session.last_use = time()
    return jsonify({"state": old_state.dict()})


def idle_session_watchdog(ttl_seconds: int = 1200):
    """Background thread to perform periodic garbage collection of sessions
    that haven't been used in `ttl_seconds` seconds.
    """
    while True:
        session_ids_to_remove = []
        for session_id, session in sessions.items():
            if session.last_use + ttl_seconds < time():
                session_ids_to_remove.append(session_id)
        with sessions_lock:
            for session_id in session_ids_to_remove:
                sessions[session_id].close()
                del sessions[session_id]
        logger.info("Garbage collected %d sessions", len(session_ids_to_remove))
        sleep(ttl_seconds)


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
    Thread(target=idle_session_watchdog).start()
    app.run(port=int(os.environ.get("PORT", "5000")))
