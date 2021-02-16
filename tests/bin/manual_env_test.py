# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for //compiler_gym/bin:manual_env."""
import pytest
from absl import app, flags
import re

import sys
from io import StringIO
from random import seed
from compiler_gym.bin.manual_env import main
from compiler_gym.util.flags.benchmark_from_flags import benchmark_from_flags
from compiler_gym.util.flags.env_from_flags import env_from_flags
from compiler_gym.util.capture_output import capture_output
from tests.test_main import main as _test_main

FLAGS = flags.FLAGS


def io_check(input, output, rnd_seed = 100):
    """Run the shell with the given input and check the output matches the
    output regex"""
    seed(rnd_seed)
    old_stdin = sys.stdin
    try:
        with capture_output() as out:
            try:
                sys.stdin = StringIO(input)
                main(["argv0", "--env=llvm-v0"])
            except SystemExit:
                pass  # Expected behaviour is to call sys.exit().
        print(out.stdout)
        
        pattern = r"""Initialized environment in [0-9.mu]*s
Welcome to the CompilerGym Shell!
---------------------------------
Type help or \? for more information. 
help tutorial will give a step by step guide.

""" + output
        assert re.match(pattern, out.stdout)
        
    finally:
        sys.stdin = old_stdin


def test_download_cBench():
    # This one needs to be called before any others
    io_check(
        """require_dataset cBench-v0""",
        r"""compilergym:NO-BENCHMARK> Downloaded dataset cBench-v0 in [0-9.mu]*s
Application must be restarted to make changes visible.""")


def test_list_datasets():
    io_check(
        """list_datasets""",
        r"""compilergym:NO-BENCHMARK> .*cBench-v0.*"""
    )


def test_list_benchmarks():
    io_check(
        """list_benchmarks""",
        r"""compilergym:NO-BENCHMARK> .*cBench-v0/adpcm.*"""
    )


def test_list_actions():
    io_check(
        """list_actions""",
        r"""compilergym:NO-BENCHMARK> .*-adce.* -strip.*"""
    )


def test_list_rewards():
    io_check(
        """list_rewards""",
        r"""compilergym:NO-BENCHMARK> .*IrInstructionCount.* ObjectTextSizeOz.*"""
    )


def test_list_observations():
    io_check(
        """list_observations""",
        r"""compilergym:NO-BENCHMARK> Autophase, .*, Programl"""
    )


def test_set_benchmark():
    io_check(
        """set_benchmark cBench-v0/adpcm""",
        r"""compilergym:NO-BENCHMARK> Reset benchmark://cBench-v0/adpcm environment in [0-9.mu]*s"""
    )


def test_actions_stack_back_stack():
    io_check(
        """set_benchmark cBench-v0/adpcm
        action - -adce -
        stack
        back
        stack""",
        r"""compilergym:NO-BENCHMARK> Reset benchmark://cBench-v0/adpcm environment in [0-9.mu]*s
compilergym:cBench-v0/adpcm> Action -rewrite-symbols
No effect
Action -adce
Action -rewrite-statepoints-for-gc
No effect
Actions -rewrite-symbols -adce -rewrite-statepoints-for-gc in [0-9.mu]*s.
compilergym:cBench-v0/adpcm>    Depth | Action                      | Effect   | Done   | Reward   |   Cumulative Reward
---------+-----------------------------+----------+--------+----------+---------------------
       3 | -rewrite-statepoints-for-gc | False    | False  | -        |                   0
       2 | -adce                       | True     | False  | -        |                   0
       1 | -rewrite-symbols            | False    | False  | -        |                   0
       0 | <init>                      | False    | False  | 0        |                   0
compilergym:cBench-v0/adpcm> Undid -rewrite-statepoints-for-gc in [0-9.mu]*s
compilergym:cBench-v0/adpcm>    Depth | Action           | Effect   | Done   | Reward   |   Cumulative Reward
---------+------------------+----------+--------+----------+---------------------
       2 | -adce            | True     | False  | -        |                   0
       1 | -rewrite-symbols | False    | False  | -        |                   0
       0 | <init>           | False    | False  | 0        |                   0"""
    )

def test_reward():
    io_check(
        """set_benchmark cBench-v0/adpcm
        set_default_reward IrInstructionCount
        action -mem2reg
        reward
        reward IrInstructionCountNorm
        stack
        """,
        r"""compilergym:NO-BENCHMARK> Reset benchmark://cBench-v0/adpcm environment in [0-9.mu]*s
compilergym:cBench-v0/adpcm> Reward IrInstructionCount in [0-9.mu]*s
compilergym:cBench-v0/adpcm> Action -mem2reg
Reward: 181.000000
Actions -mem2reg in [0-9.mu]*s.
compilergym:cBench-v0/adpcm> 0.000000
Reward IrInstructionCount in [0-9.mu]*s
compilergym:cBench-v0/adpcm> 0.000000
Reward IrInstructionCountNorm in [0-9.mu]*s
compilergym:cBench-v0/adpcm>    Depth | Action   | Effect   | Done   |   Reward |   Cumulative Reward
---------+----------+----------+--------+----------+---------------------
       1 | -mem2reg | True     | False  |      181 |                 181
       0 | <init>   | False    | False  |        0 |                   0
compilergym:cBench-v0/adpcm>    Depth | Action   | Effect   | Done   |   Reward |   Cumulative Reward
---------+----------+----------+--------+----------+---------------------
       1 | -mem2reg | True     | False  |      181 |                 181
       0 | <init>   | False    | False  |        0 |                   0"""
    )


def test_observation():
    io_check(
        """set_benchmark cBench-v0/adpcm
        set_default_observation IrInstructionCount
        action -mem2reg
        observation
        observation IrInstructionCountOz
        """,
        r"""compilergym:NO-BENCHMARK> Reset benchmark://cBench-v0/adpcm environment in [0-9.mu]*s
compilergym:cBench-v0/adpcm> Observation IrInstructionCount in [0-9.mu]*s
compilergym:cBench-v0/adpcm> Action -mem2reg
Observation: \[267\]
Actions -mem2reg in [0-9.mu]*s.
compilergym:cBench-v0/adpcm> \[267\]
Observation IrInstructionCount in [0-9.mu]*s
compilergym:cBench-v0/adpcm> \[206\]
Observation IrInstructionCountOz in [0-9.mu]*s
compilergym:cBench-v0/adpcm> \[206\]
Observation IrInstructionCountOz in [0-9.mu]*s"""
    )


def test_try_all_actions():
    io_check(
        """set_benchmark cBench-v0/adpcm
        set_default_reward IrInstructionCount
        try_all_actions""",
        r"""compilergym:NO-BENCHMARK> Reset benchmark://cBench-v0/adpcm environment in [0-9.mu]*s
compilergym:cBench-v0/adpcm> Reward IrInstructionCount in [0-9.mu]*s
compilergym:cBench-v0/adpcm> Action: -add-discriminators Reward: 0.000000
Action: -adce Reward: 1.000000
(.|\n)*
Got actions in [0-9.mu]*s
 Action                          | Effect   | Done   |   Eager Reward
---------------------------------+----------+--------+----------------
 -mem2reg                        | True     | False  |            181
 -sroa                           | True     | False  |            181
 -newgvn                         | True     | False  |             74
 -gvn                            | True     | False  |             72
(.|\n)*
 -structurizecfg                 | True     | False  |            -25
 -bounds-checking                | True     | False  |            -60"""    
    )


def test_simplify_stack():
    io_check(
        """set_benchmark cBench-v0/adpcm
        set_default_reward IrInstructionCount
        action - - - - - - - - - - - - - - -
        simplify_stack
        stack""",
        r"""compilergym:NO-BENCHMARK> Reset benchmark://cBench-v0/adpcm environment in [0-9.mu]*s
compilergym:cBench-v0/adpcm> Reward IrInstructionCount in [0-9.mu]*s
compilergym:cBench-v0/adpcm> Action -rewrite-symbols
Reward: 0.000000
No effect
Action -rewrite-statepoints-for-gc
(.|\n)*
Reward: 0.000000
Actions -rewrite-symbols -rewrite-statepoints-for-gc .* -loop-guard-widening -simplifycfg -inferattrs in [0-9.mu]*s.
compilergym:cBench-v0/adpcm> Warning previous eager reward at 13: -simplifycfg was 14.000000 now 13.000000
compilergym:cBench-v0/adpcm>    Depth | Action       | Effect   | Done   |   Reward |   Cumulative Reward
---------+--------------+----------+--------+----------+---------------------
       2 | -simplifycfg | True     | False  |       13 |                  87
       1 | -newgvn      | True     | False  |       74 |                  74
       0 | <init>       | False    | False  |        0 |                   0"""    
    )


def test_simplify_stack_no_reward():
    io_check(
        """set_benchmark cBench-v0/adpcm
        action - - - - - - - - - - - - - - -
        simplify_stack
        stack""",
        r"""compilergym:NO-BENCHMARK> Reset benchmark://cBench-v0/adpcm environment in [0-9.mu]*s
compilergym:cBench-v0/adpcm> Action -rewrite-symbols
No effect
Action -rewrite-statepoints-for-gc
(.|\n)*
Actions -rewrite-symbols -rewrite-statepoints-for-gc .* -loop-guard-widening -simplifycfg -inferattrs in [0-9.mu]*s.
compilergym:cBench-v0/adpcm> compilergym:cBench-v0/adpcm>    Depth | Action         | Effect   | Done   | Reward   |   Cumulative Reward
---------+----------------+----------+--------+----------+---------------------
       4 | -inferattrs    | True     | False  | -        |                   0
       3 | -simplifycfg   | True     | False  | -        |                   0
       2 | -functionattrs | True     | False  | -        |                   0
       1 | -newgvn        | True     | False  | -        |                   0
       0 | <init>         | False    | False  | 0        |                   0"""    
    )

def test_hill_climb():
    io_check(
        """set_benchmark cBench-v0/adpcm
        set_default_reward IrInstructionCount
        hill_climb 10
        stack""",
        r"""compilergym:NO-BENCHMARK> Reset benchmark://cBench-v0/adpcm environment in [0-9.mu]*s
compilergym:cBench-v0/adpcm> Reward IrInstructionCount in [0-9.mu]*s
compilergym:cBench-v0/adpcm> Step: 1 Action: -rewrite-symbols Reward: 0.000000 Accept: False
Step: 2 Action: -rewrite-statepoints-for-gc Reward: 0.000000 Accept: False
Step: 3 Action: -globalsplit Reward: 0.000000 Accept: False
Step: 4 Action: -newgvn Reward: 74.000000 Accept: True
Step: 5 Action: -lowerinvoke Reward: 0.000000 Accept: False
Step: 6 Action: -functionattrs Reward: 0.000000 Accept: False
Step: 7 Action: -strip-dead-prototypes Reward: 0.000000 Accept: False
Step: 8 Action: -die Reward: 0.000000 Accept: False
Step: 9 Action: -mergereturn Reward: -1.000000 Accept: False
Step: 10 Action: -div-rem-pairs Reward: 0.000000 Accept: False
Hill climb complete in [0-9.mu]*s. Accepted 1 of 10 steps for total reward of 74.0.
compilergym:cBench-v0/adpcm>    Depth | Action   | Effect   | Done   |   Reward |   Cumulative Reward
---------+----------+----------+--------+----------+---------------------
       1 | -newgvn  | True     | False  |       74 |                  74
       0 | <init>   | False    | False  |        0 |                   0"""
    )
    
def test_greedy():
    io_check(
        """set_benchmark cBench-v0/adpcm
        set_default_reward IrInstructionCount
        greedy
        stack""",
        r"""compilergym:NO-BENCHMARK> Reset benchmark://cBench-v0/adpcm environment in [0-9.mu]*s
compilergym:cBench-v0/adpcm> Reward IrInstructionCount in [0-9.mu]*s
compilergym:cBench-v0/adpcm> Action: -add-discriminators Reward: 0.000000
Action: -adce Reward: 1.000000
(.|\n)*
Action: -mem2reg Reward: 181.000000
(.|\n)*
Action: -mergereturn Reward: -1.000000
Step: 1 Selected action: -mem2reg Reward: 181.000000
Greedy 1 steps in [0-9.mu]*s
compilergym:cBench-v0/adpcm>    Depth | Action   | Effect   | Done   |   Reward |   Cumulative Reward
---------+----------+----------+--------+----------+---------------------
       1 | -mem2reg | True     | False  |      181 |                 181
       0 | <init>   | False    | False  |        0 |                   0"""
    )
    
def test_commandline():
    io_check(
        """set_benchmark cBench-v0/adpcm
        action - -
        commandline""",
        r"""compilergym:NO-BENCHMARK> Reset benchmark://cBench-v0/adpcm environment in [0-9.mu]*s
compilergym:cBench-v0/adpcm> Action -rewrite-symbols
No effect
Action -rewrite-statepoints-for-gc
No effect
Actions -rewrite-symbols -rewrite-statepoints-for-gc in [0-9.mu]*s.
compilergym:cBench-v0/adpcm> \$ opt -rewrite-symbols -rewrite-statepoints-for-gc input.bc -o output.bc"""
    )

def test_reset():
    io_check(
        """set_benchmark cBench-v0/adpcm
        action - -
        reset
        stack""",
        r"""compilergym:NO-BENCHMARK> Reset benchmark://cBench-v0/adpcm environment in [0-9.mu]*s
compilergym:cBench-v0/adpcm> Action -rewrite-symbols
No effect
Action -rewrite-statepoints-for-gc
No effect
Actions -rewrite-symbols -rewrite-statepoints-for-gc in [0-9.mu]*s.
compilergym:cBench-v0/adpcm> Reset in [0-9.mu]*s
compilergym:cBench-v0/adpcm>    Depth | Action   | Effect   | Done   |   Reward |   Cumulative Reward
---------+----------+----------+--------+----------+---------------------
       0 | <init>   | False    | False  |        0 |                   0"""
    )


def test_unrecognized_flags():
    FLAGS.unparse_flags()
    with pytest.raises(app.UsageError) as ctx:
        main(["argv0", "unknown-option"])
    assert str(ctx.value) == "Unknown command line arguments: ['unknown-option']"


def test_missing_required_flag():
    FLAGS.unparse_flags()
    with pytest.raises(app.UsageError) as ctx:
        main(["argv0"])
    assert str(ctx.value) == "Neither --env or --local_service_binary is set"


def test_ls_env():
    with capture_output() as out:
        try:
            main(["argv0", "--ls_env"])
        except SystemExit:
            pass  # Expected behaviour is to call sys.exit().
    assert "llvm-" in out.stdout


if __name__ == "__main__":
    _test_main()
