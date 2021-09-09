# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the GCC CompilerGym service."""
import gym
import numpy as np
import pytest

from compiler_gym.envs.gcc import GccEnv
from compiler_gym.service import SessionNotFound
from compiler_gym.service.connection import ServiceError
from compiler_gym.spaces import Scalar, Sequence
from tests.pytest_plugins.common import with_docker, without_docker
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.gcc"]


@without_docker
def test_gcc_env_fails_without_docker():
    with pytest.raises(ServiceError, match="Is docker installed?"):
        gym.make("gcc-v0")


@with_docker
def test_versions(env: GccEnv):
    """Tests the GetVersion() RPC endpoint."""
    assert env.compiler_version == "1.0.0"


@with_docker
def test_gcc_version(env: GccEnv):
    """Test that the gcc version is correct."""
    env.reset()
    assert env.gcc_spec.version == "gcc (GCC) 11.2.0"


@with_docker
def test_action_space(env: GccEnv):
    """Test that the environment reports the service's action spaces."""
    assert env.action_spaces[0].name == "default"
    assert len(env.action_spaces[0].names) == 2280
    assert env.action_spaces[0].names[0] == "-O0"


@with_docker
def test_observation_spaces(env: GccEnv):
    """Test that the environment reports the service's observation spaces."""
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
        min=-1, max=np.iinfo(np.int64).max, dtype=np.int
    )
    assert env.observation.spaces["asm"].space == Sequence(
        size_range=(0, None), dtype=str, opaque_data_format=""
    )


@with_docker
def test_reward_spaces(env: GccEnv):
    """Test that the environment reports the service's reward spaces."""
    env.reset()
    assert env.reward.spaces.keys() == {"asm_size", "obj_size"}


@with_docker
def test_step_before_reset(env: GccEnv):
    """Taking a step() before reset() is illegal."""
    with pytest.raises(SessionNotFound, match=r"Must call reset\(\) before step\(\)"):
        env.step(0)


@with_docker
def test_observation_before_reset(env: GccEnv):
    """Taking an observation before reset() is illegal."""
    with pytest.raises(SessionNotFound, match=r"Must call reset\(\) before step\(\)"):
        _ = env.observation["asm"]


@with_docker
def test_reward_before_reset(env: GccEnv):
    """Taking a reward before reset() is illegal."""
    with pytest.raises(SessionNotFound, match=r"Must call reset\(\) before step\(\)"):
        _ = env.reward["obj_size"]


@with_docker
def test_reset_invalid_benchmark(env: GccEnv):
    """Test requesting a specific benchmark."""
    with pytest.raises(LookupError) as ctx:
        env.reset(benchmark="chstone-v1/flubbedydubfishface")
    assert str(ctx.value) == "'benchmark://chstone-v1'"


@with_docker
def test_invalid_observation_space(env: GccEnv):
    """Test error handling with invalid observation space."""
    with pytest.raises(LookupError):
        env.observation_space = 100


@with_docker
def test_invalid_reward_space(env: GccEnv):
    """Test error handling with invalid reward space."""
    with pytest.raises(LookupError):
        env.reward_space = 100


@with_docker
def test_double_reset(env: GccEnv):
    """Test that reset() can be called twice."""
    env.reset()
    assert env.in_episode
    env.step(env.action_space.sample())
    env.reset()
    env.step(env.action_space.sample())
    assert env.in_episode


@with_docker
def test_Step_out_of_range(env: GccEnv):
    """Test error handling with an invalid action."""
    env.reset()
    with pytest.raises(ValueError) as ctx:
        env.step(10000)
    assert str(ctx.value) == "Out-of-range"


@with_docker
def test_default_benchmark(env: GccEnv):
    """Test that we are working with the expected default benchmark."""
    assert env.benchmark.proto.uri == "benchmark://chstone-v0/adpcm"


@with_docker
def test_default_reward(env: GccEnv):
    """Test default reward space."""
    env.reward_space = "obj_size"
    env.reset()
    observation, reward, done, info = env.step(0)
    assert observation is None
    assert reward == 0
    assert not done


@with_docker
def test_source_observation(env: GccEnv):
    """Test observation spaces."""
    env.reset()
    lines = env.source.split("\n")
    assert lines[0].startswith('# 0 "')
    assert lines[0].endswith('adpcm.c"')


@with_docker
def test_rtl_observation(env: GccEnv):
    """Test observation spaces."""
    env.reset()
    assert env.rtl.startswith(
        """
;; Function abs (abs, funcdef_no=0, decl_uid=1084, cgraph_uid=1, symbol_order=90)

(note 1 0 4 NOTE_INSN_DELETED)
(note 4 1 38 2 [bb 2] NOTE_INSN_BASIC_BLOCK)"""
    )


@with_docker
def test_asm_observation(env: GccEnv):
    """Test observation spaces."""
    env.reset()
    assert env.asm.startswith('\t.file\t"src.c"\n\t')


@with_docker
def test_asm_size_observation(env: GccEnv):
    """Test observation spaces."""
    env.reset()
    assert env.asm_size == 39876


@with_docker
def test_asm_hash_observation(env: GccEnv):
    """Test observation spaces."""
    env.reset()
    assert env.asm_hash == "f4921de395b026a55eab3844c8fe43dd"


@with_docker
def test_instruction_counts_observation(env: GccEnv):
    """Test observation spaces."""
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


@with_docker
def test_obj_observation(env: GccEnv):
    """Test observation spaces."""
    env.reset()
    assert env.obj[:5] == b"\x7fELF\x02"


@with_docker
def test_obj_size_observation(env: GccEnv):
    """Test observation spaces."""
    env.reset()
    assert env.obj_size == 21192


@with_docker
def test_obj_hash_observation(env: GccEnv):
    """Test observation spaces."""
    env.reset()
    assert env.obj_hash == "65937217c3758faf655df98741fe1d52"


@with_docker
def test_choices_observation(env: GccEnv):
    """Test observation spaces."""
    env.reset()
    choices = env.choices
    assert len(choices) == 502
    assert all(map(lambda x: x == -1, choices))


@with_docker
def test_command_line_observation(env: GccEnv):
    """Test observation spaces."""
    env.reset()
    command_line = env.command_line
    assert command_line == "docker:gcc:11.2.0 -w -c src.c -o obj.o"


@with_docker
def test_gcc_spec(env: GccEnv):
    """Test gcc_spec param."""
    env.reset()
    assert env.gcc_spec.gcc.bin == "docker:gcc:11.2.0"
    assert min(map(len, env.gcc_spec.options)) > 0


@with_docker
def test_set_choices(env: GccEnv):
    """Test that we can set the command line parameters"""
    env.reset()
    env.choices = [-1] * len(env.gcc_spec.options)
    assert env.command_line.startswith("docker:gcc:11.2.0 -w -c src.c -o obj.o")
    env.choices = [0] * len(env.gcc_spec.options)
    assert env.command_line.startswith(
        "docker:gcc:11.2.0 -O0 -faggressive-loop-optimizations -falign-functions -falign-jumps -falign-labels"
    )


@with_docker
def test_rewards(env: GccEnv):
    """Test reward spaces."""
    env.reset()
    assert env.reward["asm_size"] == 0
    assert env.reward["obj_size"] == 0
    env.step(env.action_space.names.index("-O3"))
    assert env.reward["asm_size"] == -19235.0
    assert env.reward["obj_size"] == -6520.0


@with_docker
def test_timeout(env: GccEnv):
    """Test that the timeout can be set. Can't really make it timeout, I think."""
    env.reset()
    env.timeout = 20
    assert env.timeout == 20
    env.reset()
    assert env.timeout == 20


@with_docker
def test_benchmarks(env: GccEnv):
    assert list(env.datasets.benchmark_uris())[0] == "benchmark://chstone-v0/adpcm"


@with_docker
def test_compile(env: GccEnv):
    env.observation_space = "obj_size"
    observation = env.reset()
    assert observation == 21192
    observation, _, _, _ = env.step(env.action_space.names.index("-O0"))
    assert observation == 21192
    observation, _, _, _ = env.step(env.action_space.names.index("-O3"))
    assert observation == 27712
    observation, _, _, _ = env.step(env.action_space.names.index("-finline"))
    assert observation == 27712


@with_docker
def test_fork(env: GccEnv):
    env.reset()
    env.step(0)
    env.step(1)
    other_env = env.fork()
    try:
        assert env.benchmark == other_env.benchmark
        assert other_env.actions == [0, 1]
    finally:
        other_env.close()


if __name__ == "__main__":
    main()
