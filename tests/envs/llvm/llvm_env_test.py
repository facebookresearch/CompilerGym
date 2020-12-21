# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the LLVM CompilerGym environments."""
import os
import sys
from typing import Any, Dict, List

import networkx as nx
import numpy as np
import pytest
from gym.spaces import Box
from gym.spaces import Dict as DictSpace

from compiler_gym.envs import CompilerEnv
from compiler_gym.envs.llvm.llvm_env import LlvmEnv
from compiler_gym.spaces import Sequence
from tests.test_main import main

pytest_plugins = ["tests.envs.llvm.fixtures"]


def test_eager_reward_space(env: CompilerEnv):
    env.eager_reward_space = "IrInstructionCount"
    assert env.eager_reward_space == "IrInstructionCount"

    env.eager_reward_space = None
    assert env.eager_reward_space is None

    invalid = "invalid value"
    with pytest.raises(LookupError) as ctx:
        env.eager_reward_space = invalid
    assert str(ctx.value) == f"Reward space not found: {invalid}"


def test_eager_observation_space(env: CompilerEnv):
    env.eager_observation_space = "Autophase"
    assert env.eager_observation_space == "Autophase"

    env.eager_observation_space = None
    assert env.eager_observation_space is None

    invalid = "invalid value"
    with pytest.raises(LookupError) as ctx:
        env.eager_observation_space = invalid
    assert str(ctx.value) == f"Observation space not found: {invalid}"


def test_action_space_names(env: CompilerEnv, action_names: List[str]):
    assert set(env.action_space.names) == set(action_names)


def test_action_spaces_names(env: CompilerEnv):
    assert {a.name for a in env.action_spaces} == {"PassesAll"}


def test_all_flags_are_unique(env: LlvmEnv):
    assert sorted(env.action_space.flags) == sorted(set(env.action_space.flags))


def test_benchmark_names(env: CompilerEnv, benchmark_names: List[str]):
    assert set(benchmark_names) == set(env.benchmarks)


def test_observation_spaces(env: CompilerEnv):
    env.reset("cBench-v0/crc32")

    assert set(env.observation.spaces.keys()) == {
        "Ir",
        "BitcodeFile",
        "Autophase",
        "AutophaseDict",
        "Programl",
        "CpuInfo",
        "CpuInfoDict",
        "Inst2vecPreprocessedText",
        "Inst2vecEmbeddingIndices",
        "Inst2vec",
        "IrInstructionCount",
        "IrInstructionCountO0",
        "IrInstructionCountO3",
        "IrInstructionCountOz",
        "NativeTextSizeBytes",
        "NativeTextSizeBytesO0",
        "NativeTextSizeBytesO3",
        "NativeTextSizeBytesOz",
    }


def test_ir_observation_space(env: CompilerEnv):
    env.reset("cBench-v0/crc32")
    key = "Ir"
    space = env.observation.spaces[key]
    assert isinstance(space, Sequence)
    assert space.dtype == str
    assert space.size_range == (0, None)

    value: str = env.observation[key]
    assert isinstance(value, str)
    assert space.contains(value)


def test_bitcode_observation_space(env: CompilerEnv):
    env.reset("cBench-v0/crc32")
    key = "BitcodeFile"
    space = env.observation.spaces[key]
    assert isinstance(space, Sequence)
    assert space.dtype == str
    assert space.size_range == (0, 4096)

    value: str = env.observation[key]
    try:
        assert isinstance(value, str)
        assert os.path.isfile(value)
        assert space.contains(value)
    finally:
        os.unlink(value)


def test_autophase_observation_space(env: CompilerEnv):
    env.reset("cBench-v0/crc32")
    key = "Autophase"
    space = env.observation.spaces[key]
    assert isinstance(space, Box)

    value: np.ndarray = env.observation[key]
    assert isinstance(value, np.ndarray)
    assert value.shape == (56,)

    np.testing.assert_array_equal(
        value,
        [
            0,
            0,
            16,
            12,
            2,
            16,
            8,
            2,
            4,
            8,
            0,
            0,
            0,
            29,
            0,
            24,
            9,
            2,
            32,
            38,
            21,
            14,
            30,
            16,
            13,
            0,
            5,
            24,
            3,
            3,
            26,
            0,
            24,
            13,
            5,
            10,
            3,
            51,
            0,
            1,
            0,
            5,
            0,
            0,
            0,
            38,
            0,
            1,
            8,
            5,
            29,
            196,
            131,
            13,
            0,
            81,
        ],
    )

    assert space.contains(value)


def test_autophase_dict_observation_space(env: CompilerEnv):
    env.reset("cBench-v0/crc32")
    key = "AutophaseDict"
    space = env.observation.spaces[key]
    assert isinstance(space, DictSpace)
    value: Dict[str, int] = env.observation[key]
    assert len(value) == 56


def test_programl_observation_space(env: CompilerEnv):
    env.reset("cBench-v0/crc32")
    key = "Programl"
    assert isinstance(env.observation.spaces[key], Sequence)
    graph: nx.MultiDiGraph = env.observation[key]
    assert isinstance(graph, nx.MultiDiGraph)

    assert graph.number_of_nodes() == 419
    assert graph.number_of_edges() == 703
    assert graph.nodes[0] == {
        "block": 0,
        "function": 0,
        "text": "[external]",
        "type": 0,
    }


def test_cpuinfo_observation_space(env: CompilerEnv):
    env.reset("cBench-v0/crc32")
    space = "CpuInfo"
    assert isinstance(env.observation.spaces[space], Sequence)
    value: Dict[str, Any] = env.observation[space]
    assert isinstance(value, dict)
    # Test each expected key, removing it as we go.
    assert isinstance(value.pop("name"), str)
    assert isinstance(value.pop("cores_count"), int)
    assert isinstance(value.pop("l1i_cache_size"), int)
    assert isinstance(value.pop("l1i_cache_count"), int)
    assert isinstance(value.pop("l1d_cache_size"), int)
    assert isinstance(value.pop("l1d_cache_count"), int)
    assert isinstance(value.pop("l2_cache_size"), int)
    assert isinstance(value.pop("l2_cache_count"), int)
    assert isinstance(value.pop("l3_cache_size"), int)
    assert isinstance(value.pop("l3_cache_count"), int)
    assert isinstance(value.pop("l4_cache_size"), int)
    assert isinstance(value.pop("l4_cache_count"), int)
    # Anything left in the JSON dictionary now is an unexpected key.
    assert not value

    invalid = "invalid value"
    with pytest.raises(KeyError) as ctx:
        _ = env.observation[invalid]
    assert str(ctx.value) == f"'{invalid}'"

    space = "CpuInfoDict"
    value: Dict[str, Any] = env.observation[space]
    print(value)
    # assert env.observation.spaces[space].contains(value)


@pytest.fixture
def cbench_crc32_inst2vec_embedding_indices() -> List[int]:
    """The expected inst2vec embedding indices for cBench-v0/crc32."""
    # The linux/macOS builds of clang produce slightly different bitcodes.
    if sys.platform.lower().startswith("linux"):
        return [
            8564,
            8564,
            5,
            46,
            46,
            40,
            8564,
            13,
            8,
            8564,
            1348,
            178,
            286,
            214,
            182,
            235,
            697,
            1513,
            192,
            8564,
            182,
            182,
            395,
            1513,
            2298,
            8564,
            289,
            291,
            3729,
            3729,
            8564,
            178,
            289,
            200,
            1412,
            1412,
            205,
            415,
            205,
            213,
            8564,
            8564,
            5666,
            204,
            8564,
            213,
            215,
            364,
            216,
            8564,
            216,
            8564,
            5665,
            8564,
            311,
            634,
            204,
            8564,
            415,
            182,
            640,
            214,
            182,
            295,
            675,
            697,
            1513,
            192,
            8564,
            182,
            182,
            395,
            1513,
            214,
            216,
            8564,
            5665,
            8564,
            634,
            204,
            8564,
            213,
            215,
            415,
            205,
            216,
            8564,
            5665,
            8564,
            182,
            961,
            214,
            415,
            214,
            364,
            216,
            8564,
            295,
            257,
            8564,
            291,
            178,
            178,
            200,
            214,
            205,
            216,
            8564,
            182,
            977,
            204,
            8564,
            182,
            213,
            235,
            697,
            1513,
            192,
            8564,
            182,
            182,
            395,
            1513,
            214,
            216,
            8564,
            182,
            420,
            214,
            213,
            8564,
            200,
            216,
            8564,
            182,
            961,
            2298,
            8564,
            289,
            8564,
            289,
            178,
            178,
            289,
            311,
            594,
            311,
            364,
            216,
            8564,
            295,
            431,
            311,
            425,
            204,
            8564,
            597,
            8564,
            594,
            213,
            8564,
            295,
            653,
            311,
            295,
            634,
            204,
            8564,
            182,
            182,
            597,
            213,
            8564,
            216,
            8564,
            216,
            8564,
            295,
            634,
            612,
            257,
            8564,
            289,
            289,
            8564,
            8564,
            178,
            178,
            364,
            311,
            594,
            8564,
            8564,
            8564,
            5666,
            204,
            8564,
            5391,
            8564,
            364,
            216,
            8564,
            5665,
            8564,
            5665,
            8564,
            205,
            216,
            8564,
            182,
            182,
            488,
            204,
            8564,
            295,
            597,
            182,
            640,
            182,
            540,
            612,
            8564,
            216,
            8564,
            182,
            640,
            214,
            216,
            8564,
            364,
            216,
            8564,
            295,
            257,
        ]
    elif sys.platform.lower().startswith("darwin"):
        return [
            8564,
            8564,
            5,
            46,
            46,
            40,
            8564,
            13,
            8,
            8564,
            1348,
            178,
            286,
            214,
            182,
            235,
            697,
            1513,
            192,
            8564,
            182,
            182,
            395,
            1513,
            2298,
            8564,
            289,
            291,
            3729,
            3729,
            8564,
            178,
            289,
            200,
            1412,
            1412,
            205,
            415,
            205,
            213,
            8564,
            8564,
            5666,
            204,
            8564,
            213,
            215,
            364,
            216,
            8564,
            216,
            8564,
            5665,
            8564,
            311,
            634,
            204,
            8564,
            415,
            182,
            640,
            214,
            182,
            295,
            675,
            697,
            1513,
            192,
            8564,
            182,
            182,
            395,
            1513,
            214,
            216,
            8564,
            5665,
            8564,
            634,
            204,
            8564,
            213,
            215,
            415,
            205,
            216,
            8564,
            5665,
            8564,
            182,
            961,
            214,
            415,
            214,
            364,
            216,
            8564,
            295,
            257,
            8564,
            291,
            178,
            178,
            200,
            214,
            205,
            216,
            8564,
            182,
            977,
            204,
            8564,
            182,
            213,
            235,
            697,
            1513,
            192,
            8564,
            182,
            182,
            395,
            1513,
            214,
            216,
            8564,
            182,
            420,
            214,
            213,
            8564,
            200,
            216,
            8564,
            182,
            961,
            2298,
            8564,
            289,
            8564,
            289,
            178,
            178,
            289,
            311,
            594,
            311,
            364,
            216,
            8564,
            295,
            431,
            311,
            425,
            204,
            8564,
            597,
            8564,
            594,
            213,
            8564,
            295,
            653,
            311,
            295,
            634,
            204,
            8564,
            182,
            182,
            597,
            213,
            8564,
            216,
            8564,
            216,
            8564,
            295,
            634,
            612,
            257,
            8564,
            289,
            289,
            8564,
            8564,
            178,
            178,
            364,
            311,
            594,
            8564,
            8564,
            8564,
            5666,
            204,
            8564,
            5391,
            8564,
            364,
            216,
            8564,
            5665,
            8564,
            5665,
            8564,
            205,
            216,
            8564,
            182,
            182,
            488,
            204,
            8564,
            295,
            597,
            182,
            640,
            182,
            540,
            612,
            8564,
            216,
            8564,
            182,
            640,
            214,
            216,
            8564,
            364,
            216,
            8564,
            295,
            257,
        ]
    else:
        raise NotImplementedError(f"Unknown platform: {sys.platform}")


def test_inst2vec_preprocessed_observation_space(
    env: CompilerEnv, cbench_crc32_inst2vec_embedding_indices: List[int]
):
    env.reset("cBench-v0/crc32")
    space = "Inst2vecPreprocessedText"
    assert isinstance(env.observation.spaces[space], Sequence)
    value: List[str] = env.observation[space]

    assert isinstance(value, list)
    for item, idx in zip(value, cbench_crc32_inst2vec_embedding_indices):
        assert isinstance(item, str)
    unk = env.observation.inst2vec.vocab["!UNK"]
    indices = [env.observation.inst2vec.vocab.get(item, unk) for item in value]
    assert indices == cbench_crc32_inst2vec_embedding_indices


def test_inst2vec_embedding_indices_observation_space(
    env: CompilerEnv, cbench_crc32_inst2vec_embedding_indices: List[int]
):
    env.reset("cBench-v0/crc32")
    space = "Inst2vecEmbeddingIndices"
    assert isinstance(env.observation.spaces[space], Sequence)
    value: List[int] = env.observation[space]

    print(value)
    assert isinstance(value, list)
    for item in value:
        assert isinstance(item, int)
    assert value == cbench_crc32_inst2vec_embedding_indices


def test_inst2vec_observation_space(
    env: CompilerEnv, cbench_crc32_inst2vec_embedding_indices: List[int]
):
    env.reset("cBench-v0/crc32")
    space = "Inst2vec"
    assert isinstance(env.observation.spaces[space], Sequence)
    value: np.ndarray = env.observation[space]

    assert isinstance(value, np.ndarray)
    assert value.dtype == np.float32
    height, width = value.shape
    assert width == len(env.observation.inst2vec.embeddings[0])
    assert height == len(cbench_crc32_inst2vec_embedding_indices)
    # Check a handful of values.
    np.testing.assert_array_almost_equal(
        value.tolist(),
        [
            env.observation.inst2vec.embeddings[idx]
            for idx in cbench_crc32_inst2vec_embedding_indices
        ],
    )


def test_ir_instruction_count_observation_space(env: CompilerEnv):
    env.reset("cBench-v0/crc32")
    key = "IrInstructionCount"
    space = env.observation.spaces[key]
    assert isinstance(space, Box)

    value: np.ndarray = env.observation[key]
    assert isinstance(value, np.ndarray)
    assert value.shape == (1,)

    np.testing.assert_array_equal([196], value)


def test_reward_spaces(env: CompilerEnv):
    env.reset("cBench-v0/crc32")

    assert set(env.reward.ranges.keys()) == {
        "IrInstructionCount",
        "IrInstructionCountO3",
        "IrInstructionCountOz",
        "NativeTextSizeBytes",
        "NativeTextSizeBytesO3",
        "NativeTextSizeBytesOz",
    }

    reward_space = "IrInstructionCount"
    assert env.reward.ranges[reward_space] == (-np.inf, np.inf)
    assert env.reward[reward_space] == 0

    reward_space = "IrInstructionCountO3"
    assert env.reward.ranges[reward_space] == (-np.inf, np.inf)
    assert env.reward[reward_space] == 0

    reward_space = "IrInstructionCountOz"
    assert env.reward.ranges[reward_space] == (-np.inf, np.inf)
    assert env.reward[reward_space] == 0

    reward_space = "NativeTextSizeBytes"
    assert env.reward.ranges[reward_space] == (-np.inf, np.inf)
    assert env.reward[reward_space] == 0

    reward_space = "NativeTextSizeBytesO3"
    assert env.reward.ranges[reward_space] == (-np.inf, np.inf)
    assert env.reward[reward_space] == 0

    reward_space = "NativeTextSizeBytesOz"
    assert env.reward.ranges[reward_space] == (-np.inf, np.inf)
    assert env.reward[reward_space] == 0

    invalid = "invalid value"
    with pytest.raises(KeyError) as ctx:
        _ = env.reward[invalid]
    assert str(ctx.value) == f"'{invalid}'"


def test_double_reset(env: CompilerEnv):
    env.reset(benchmark="cBench-v0/crc32")
    env.reset(benchmark="cBench-v0/crc32")
    assert env.in_episode


def test_service_env_dies_reset(env: CompilerEnv):
    env.eager_observation_space = "Autophase"
    env.eager_reward_space = "IrInstructionCount"
    env.reset("cBench-v0/crc32")

    # Kill the service.
    env.service.close()

    # Check that the environment doesn't fall over.
    observation, reward, done, _ = env.step(0)
    assert done
    assert observation is None
    assert reward is None

    # Reset the environment and check that it works.
    env.reset(benchmark="cBench-v0/crc32")
    observation, reward, done, _ = env.step(0)
    assert not done
    assert observation is not None
    assert reward is not None


def test_commandline(env: CompilerEnv):
    env.reset("cBench-v0/crc32")
    assert env.commandline() == "opt  input.bc -o output.bc"


def test_uri_substring_candidate_match(env: CompilerEnv):
    env.reset(benchmark="benchmark://cBench-v0/crc32")
    assert env.benchmark == "benchmark://cBench-v0/crc32"

    env.reset(benchmark="benchmark://cBench-v0/crc3")
    assert env.benchmark == "benchmark://cBench-v0/crc32"

    env.reset(benchmark="benchmark://cBench-v0/cr")
    assert env.benchmark == "benchmark://cBench-v0/crc32"


def test_uri_substring_candidate_match_inferref_prefix(env: CompilerEnv):
    env.reset(benchmark="cBench-v0/crc32")
    assert env.benchmark == "benchmark://cBench-v0/crc32"

    env.reset(benchmark="cBench-v0/crc3")
    assert env.benchmark == "benchmark://cBench-v0/crc32"

    env.reset(benchmark="cBench-v0/cr")
    assert env.benchmark == "benchmark://cBench-v0/crc32"


def test_reset_to_force_benchmark(env: CompilerEnv):
    """Reset that calling reset() with a benchmark forces that benchmark to
    be used for every subsequent episode.
    """
    env.benchmark = None
    env.reset(benchmark="benchmark://cBench-v0/crc32")
    assert env.benchmark == "benchmark://cBench-v0/crc32"
    for _ in range(10):
        env.reset()
        assert env.benchmark == "benchmark://cBench-v0/crc32"


def test_unset_forced_benchmark(env: CompilerEnv):
    """Test that setting benchmark to None "unsets" the user benchmark for
    every subsequent episode.
    """
    env.reset(benchmark="benchmark://cBench-v0/crc32")
    assert env.benchmark == "benchmark://cBench-v0/crc32"
    env.benchmark = None
    for _ in range(50):
        env.reset()
        if env.benchmark != "benchmark://cBench-v0/crc32":
            break
    else:
        pytest.fail(
            "Improbably selected the same benchmark 50 times! " "Expected random."
        )


if __name__ == "__main__":
    main()
