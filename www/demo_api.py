"""Demonstration of wrapping CompilerGym in a simple Flask app.

This exposes an API with three operations:

   1. start() -> session_id, state   (/api/v1/start)

        Start a session. This would happen when the user navigates to the page
        in their web browser. One tab = one session. Returns a numeric session
        ID (this probably isn't the right way of doing things but I don't know
        any better :-) ). Also returns a state, which is the set of things we
        want to visualize to represent the current environment state.

   2. step(session_id, action) -> state  (/api/v1/<session_id>/<action>)

        Run an action and produce a new state, replacing the old one.

   3. stop(session_id)  (/api/v1/stop/<session_id>)

        End a session. This would be when the user closes the tab / disconnects.

To run this script, install the python dependencies using:

    pip install flask compiler_gym pydantic

Then launch it by running, in this directory:

    FLASK_APP=demo_api.py flask run

Interact with the API through GET requests. E.g. using curl start a session,
producing an initial state and a new session ID:

    $ curl -s localhost:5000/api/v1/start | jq
    {
        "session_id": 0,
        "state": {
            "autophase": [
                0,
                4,
                54,
                39,
            12
            ],
            "codesize_reward": 0,
            "commandline": "opt  input.bc -o output.bc",
            "done": false,
            "instcount": [
                638,
                85,
                16,
                6,
                77
            ],
            "ir": "; ModuleID = '-'\nsource_filename = \"-\"\ntarget datalayout = \"e-m:o-p270:32:32-p27"
        }
    }

That "state" dict contains the things that we would want to visualize in the
GUI. Our session ID is 0, lets take a step in this session using action "10":

    $ curl -s localhost:5000/api/v1/step/0/10 | jq
    {
        "state": {
            "autophase": [
                0,
                1,
                38,
                26,
                9
            ],
            "codesize_reward": 0.06501547987616099,
            "commandline": "opt -simplifycfg input.bc -o output.bc",
            "done": false,
            "instcount": [
                617,
                67,
                16,
                6,
                59
            ],
            "ir": "; ModuleID = '-'\nsource_filename = \"-\"\ntarget datalayout = \"e-m:o-p270:32:32-p27"
        }
    }

Notice that the state dict has changed. Some of the numbers in the "autophase"
and "instcount" vectors have changed, there is a codesize_reward value, and the
commandline now includes the flag needed to run action "10" (which turned out to
be the "-simplifycfg" flag).

We could carry on taking steps, or just end the session:

    $ curl -s localhost:5000/api/v1/stop/0
"""
from typing import Dict, List

from flask import Flask, jsonify
from pydantic import BaseModel

import compiler_gym
from compiler_gym import CompilerEnv

app = Flask("compiler_gym")

# A set of sessions that are in use, keyed by a numeric session ID. This is
# almost certainly not the right way of doing things ;-)
sessions: Dict[int, CompilerEnv] = {}


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
    instcount: List[int]
    autophase: List[int]

    # The reward signal measures how "good" the previous action was. Over time
    # the sequence of actions that produces the highest cumulative reward is the
    # best:
    codesize_reward: float


def compute_state(env: CompilerEnv, actions: List[int]) -> StateToVisualize:
    """Apply a list of actions and produce a new state to visualize."""  # This is where we get the compiler environment to do its thing, and compute
    # for us all of the features that we would like to visualize.
    (ir, instcount, autophase), (codesize_reward,), done, _ = env.raw_step(
        actions=actions,
        observations=[
            env.observation.spaces["Ir"],
            env.observation.spaces["InstCount"],
            env.observation.spaces["Autophase"],
        ],
        rewards=[env.reward.spaces["IrInstructionCountOz"]],
    )
    return StateToVisualize(
        commandline=env.commandline(),
        done=done,
        # For the
        ir=ir[:80],
        instcount=instcount.tolist()[:5],
        autophase=autophase.tolist()[:5],
        codesize_reward=codesize_reward,
    )


@app.route("/api/v1/start")
def start():
    # For the sake of simplicity we're ignoring the fact that the user may want
    # to select their own "benchmark" value:
    env = compiler_gym.make("llvm-v0", benchmark="cbench-v1/qsort")
    env.reset()
    state = compute_state(env, [])
    session_id = len(sessions)
    sessions[session_id] = env
    return jsonify({"session_id": session_id, "state": state.dict()})


@app.route("/api/v1/stop/<session_id>")
def stop(session_id: int):
    session_id = int(session_id)

    sessions[session_id].close()
    del sessions[session_id]

    return jsonify({"session_id": session_id})


@app.route("/api/v1/step/<session_id>/<action>")
def step(session_id: int, action: int):
    session_id = int(session_id)
    action = int(action)

    new_state = compute_state(sessions[session_id], [action])

    return jsonify({"state": new_state.dict()})
