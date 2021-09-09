# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the GCC CompilerGym service."""
import gym
import numpy as np
import pytest

import compiler_gym.envs.gcc  # noqa register environments
from compiler_gym.service import SessionNotFound
from compiler_gym.service.connection import ServiceError
from compiler_gym.spaces import Scalar, Sequence
from tests.pytest_plugins.gcc import with_gcc_support, without_gcc_support
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.gcc"]


@without_gcc_support
def test_gcc_env_fails_without_gcc_support():
    with pytest.raises(ServiceError):
        gym.make("gcc-v0")


@with_gcc_support
def test_action_space():
    """Test that the environment reports the service's action spaces."""
    with gym.make("gcc-v0") as env:
        assert env.action_spaces[0].name == "default"
        assert len(env.action_spaces[0].names) == 2280
        assert env.action_spaces[0].names[0] == "-O0"


@with_gcc_support
def test_observation_spaces():
    """Test that the environment reports the service's observation spaces."""
    with gym.make("gcc-v0") as env:
        env.reset()
        assert env.observation.spaces.keys() == {
            "asm_hash",
            "asm_size",
            "asm",
            "choices",
            "command_line",
            "instruction_counts",
            "obj_hash",
            "obj_size",
            "obj",
            "rtl",
            "source",
        }
        assert env.observation.spaces["obj_size"].space == Scalar(
            min=-1, max=np.iinfo(np.int64).max, dtype=int
        )
        assert env.observation.spaces["asm"].space == Sequence(
            size_range=(0, None), dtype=str, opaque_data_format=""
        )


@with_gcc_support
def test_reward_spaces():
    """Test that the environment reports the service's reward spaces."""
    with gym.make("gcc-v0") as env:
        env.reset()
        assert env.reward.spaces.keys() == {"asm_size", "obj_size"}


@with_gcc_support
def test_step_before_reset():
    """Taking a step() before reset() is illegal."""
    with gym.make("gcc-v0") as env:
        with pytest.raises(
            SessionNotFound, match=r"Must call reset\(\) before step\(\)"
        ):
            env.step(0)


@with_gcc_support
def test_observation_before_reset():
    """Taking an observation before reset() is illegal."""
    with gym.make("gcc-v0") as env:
        with pytest.raises(
            SessionNotFound, match=r"Must call reset\(\) before step\(\)"
        ):
            _ = env.observation["asm"]


@with_gcc_support
def test_reward_before_reset():
    """Taking a reward before reset() is illegal."""
    with gym.make("gcc-v0") as env:
        with pytest.raises(
            SessionNotFound, match=r"Must call reset\(\) before step\(\)"
        ):
            _ = env.reward["obj_size"]


@with_gcc_support
def test_reset_invalid_benchmark():
    """Test requesting a specific benchmark."""
    with gym.make("gcc-v0") as env:
        with pytest.raises(LookupError, match=r"'benchmark://chstone-v1"):
            env.reset(benchmark="chstone-v1/flubbedydubfishface")


@with_gcc_support
def test_invalid_observation_space():
    """Test error handling with invalid observation space."""
    with gym.make("gcc-v0") as env:
        with pytest.raises(LookupError):
            env.observation_space = 100


@with_gcc_support
def test_invalid_reward_space():
    """Test error handling with invalid reward space."""
    with gym.make("gcc-v0") as env:
        with pytest.raises(LookupError):
            env.reward_space = 100


@with_gcc_support
def test_double_reset():
    """Test that reset() can be called twice."""
    with gym.make("gcc-v0") as env:
        env.reset()
        assert env.in_episode
        env.step(env.action_space.sample())
        env.reset()
        env.step(env.action_space.sample())
        assert env.in_episode


@with_gcc_support
def test_step_out_of_range():
    """Test error handling with an invalid action."""
    with gym.make("gcc-v0") as env:
        env.reset()
        with pytest.raises(ValueError, match="Out-of-range"):
            env.step(10000)


@with_gcc_support
def test_default_benchmark():
    """Test that we are working with the expected default benchmark."""
    with gym.make("gcc-v0") as env:
        assert env.benchmark.proto.uri == "benchmark://chstone-v0/adpcm"


@with_gcc_support
def test_default_reward():
    """Test default reward space."""
    with gym.make("gcc-v0") as env:
        env.reward_space = "obj_size"
        env.reset()
        observation, reward, done, info = env.step(0)
        assert observation is None
        assert reward == 0
        assert not done


@with_gcc_support
def test_source_observation():
    """Test observation spaces."""
    with gym.make("gcc-v0") as env:
        env.reset()
        lines = env.source.split("\n")
        assert lines[0].startswith('# 0 "')
        assert lines[0].endswith('adpcm.c"')


@with_gcc_support
def test_rtl_observation():
    """Test observation spaces."""
    with gym.make("gcc-v0") as env:
        env.reset()
        assert env.rtl.startswith(
            """
;; Function abs (abs, funcdef_no=0, decl_uid=1084, cgraph_uid=1, symbol_order=90)

(note 1 0 4 NOTE_INSN_DELETED)
(note 4 1 38 2 [bb 2] NOTE_INSN_BASIC_BLOCK)"""
        )


@with_gcc_support
def test_asm_observation():
    """Test observation spaces."""
    with gym.make("gcc-v0") as env:
        env.reset()
        assert env.asm.startswith('\t.file\t"src.c"\n\t')


@with_gcc_support
def test_asm_size_observation():
    """Test observation spaces."""
    with gym.make("gcc-v0") as env:
        env.reset()
        assert env.asm_size == 39876


@with_gcc_support
def test_asm_hash_observation():
    """Test observation spaces."""
    with gym.make("gcc-v0") as env:
        env.reset()
        assert env.asm_hash == "f4921de395b026a55eab3844c8fe43dd"


@with_gcc_support
def test_instruction_counts_observation():
    """Test observation spaces."""
    with gym.make("gcc-v0") as env:
        env.reset()
        assert env.instruction_counts == {
            ".align": 95,
            ".bss": 8,
            ".cfi": 91,
            ".file": 1,
            ".globl": 110,
            ".ident": 1,
            ".long": 502,
            ".section": 10,
            ".size": 110,
            ".string": 1,
            ".text": 4,
            ".type": 110,
            ".zero": 83,
            "addl": 44,
            "addq": 17,
            "andl": 2,
            "call": 34,
            "cltq": 67,
            "cmovns": 2,
            "cmpl": 30,
            "cmpq": 1,
            "imulq": 27,
            "je": 2,
            "jge": 3,
            "jle": 21,
            "jmp": 24,
            "jne": 1,
            "jns": 2,
            "js": 7,
            "leaq": 40,
            "leave": 4,
            "movl": 575,
            "movq": 150,
            "movslq": 31,
            "negl": 5,
            "negq": 1,
            "nop": 7,
            "orl": 1,
            "popq": 11,
            "pushq": 16,
            "ret": 15,
            "sall": 2,
            "salq": 7,
            "sarl": 9,
            "sarq": 20,
            "shrl": 2,
            "subl": 7,
            "subq": 15,
            "testl": 1,
            "testq": 4,
        }


@with_gcc_support
def test_obj_observation():
    """Test observation spaces."""
    with gym.make("gcc-v0") as env:
        env.reset()
        assert env.obj[:5] == b"\x7fELF\x02"


@with_gcc_support
def test_obj_size_observation():
    """Test observation spaces."""
    with gym.make("gcc-v0") as env:
        env.reset()
        assert env.obj_size == 21192


@with_gcc_support
def test_obj_hash_observation():
    """Test observation spaces."""
    with gym.make("gcc-v0") as env:
        env.reset()
        assert env.obj_hash == "65937217c3758faf655df98741fe1d52"


@with_gcc_support
def test_choices_observation():
    """Test observation spaces."""
    with gym.make("gcc-v0") as env:
        env.reset()
        choices = env.choices
        assert len(choices) == 502
        assert all(map(lambda x: x == -1, choices))


@with_gcc_support
def test_command_line_observation():
    """Test observation spaces."""
    with gym.make("gcc-v0") as env:
        env.reset()
        command_line = env.command_line
        assert command_line == "docker:gcc:11.2.0 -w -c src.c -o obj.o"


@with_gcc_support
def test_gcc_spec():
    """Test gcc_spec param."""
    with gym.make("gcc-v0") as env:
        env.reset()
        assert env.gcc_spec.gcc.bin == "docker:gcc:11.2.0"
        assert min(map(len, env.gcc_spec.options)) > 0


@with_gcc_support
def test_set_choices():
    """Test that we can set the command line parameters"""
    with gym.make("gcc-v0") as env:
        env.reset()
        env.choices = [-1] * len(env.gcc_spec.options)
        assert env.command_line.startswith("docker:gcc:11.2.0 -w -c src.c -o obj.o")
        env.choices = [0] * len(env.gcc_spec.options)
        assert env.command_line.startswith(
            "docker:gcc:11.2.0 -O0 -faggressive-loop-optimizations -falign-functions -falign-jumps -falign-labels"
        )


@with_gcc_support
def test_rewards():
    """Test reward spaces."""
    with gym.make("gcc-v0") as env:
        env.reset()
        assert env.reward["asm_size"] == 0
        assert env.reward["obj_size"] == 0
        env.step(env.action_space.names.index("-O3"))
        assert env.reward["asm_size"] == -19235.0
        assert env.reward["obj_size"] == -6520.0


@with_gcc_support
def test_timeout():
    """Test that the timeout can be set."""
    with gym.make("gcc-v0") as env:
        env.reset()
        env.timeout = 20
        assert env.timeout == 20
        env.reset()
        assert env.timeout == 20


@with_gcc_support
def test_compile():
    with gym.make("gcc-v0") as env:
        env.observation_space = "obj_size"
        observation = env.reset()
        assert observation == 21192
        observation, _, _, _ = env.step(env.action_space.names.index("-O0"))
        assert observation == 21192
        observation, _, _, _ = env.step(env.action_space.names.index("-O3"))
        assert observation == 27712
        observation, _, _, _ = env.step(env.action_space.names.index("-finline"))
        assert observation == 27712


@with_gcc_support
def test_fork():
    with gym.make("gcc-v0") as env:
        env.reset()
        env.step(0)
        env.step(1)
        fkd = env.fork()
        try:
            assert env.benchmark == fkd.benchmark
            assert fkd.actions == [0, 1]
            fkd.step(0)
            assert fkd.actions == [0, 1, 0]
            assert env.actions == [0, 1]
        finally:
            fkd.close()


if __name__ == "__main__":
    main()
