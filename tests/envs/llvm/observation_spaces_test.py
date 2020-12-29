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

from compiler_gym.envs.llvm.llvm_env import LlvmEnv
from compiler_gym.spaces import Sequence
from tests.test_main import main

pytest_plugins = ["tests.envs.llvm.fixtures"]


def test_eager_observation_space(env: LlvmEnv):
    env.observation_space = "Autophase"
    assert env.observation_space.id == "Autophase"

    env.observation_space = None
    assert env.observation_space is None

    invalid = "invalid value"
    with pytest.raises(LookupError) as ctx:
        env.observation_space = invalid
    assert str(ctx.value) == f"Observation space not found: {invalid}"


def test_observation_spaces(env: LlvmEnv):
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
        "ObjectTextSizeBytes",
        "ObjectTextSizeO0",
        "ObjectTextSizeO3",
        "ObjectTextSizeOz",
    }


def test_ir_observation_space(env: LlvmEnv):
    env.reset("cBench-v0/crc32")
    key = "Ir"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Sequence)
    assert space.space.dtype == str
    assert space.space.size_range == (0, None)

    value: str = env.observation[key]
    assert isinstance(value, str)
    assert space.space.contains(value)

    assert space.deterministic
    assert not space.platform_dependent


def test_bitcode_observation_space(env: LlvmEnv):
    env.reset("cBench-v0/crc32")
    key = "BitcodeFile"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Sequence)
    assert space.space.dtype == str
    assert space.space.size_range == (0, 4096)

    value: str = env.observation[key]
    try:
        assert isinstance(value, str)
        assert os.path.isfile(value)
        assert space.space.contains(value)
    finally:
        os.unlink(value)

    assert not space.deterministic
    assert not space.platform_dependent


def test_autophase_observation_space(env: LlvmEnv):
    env.reset("cBench-v0/crc32")
    key = "Autophase"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Box)

    value: np.ndarray = env.observation[key]
    assert isinstance(value, np.ndarray)
    assert value.shape == (56,)

    assert space.deterministic
    assert not space.platform_dependent

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

    assert space.space.contains(value)


def test_autophase_dict_observation_space(env: LlvmEnv):
    env.reset("cBench-v0/crc32")
    key = "AutophaseDict"
    space = env.observation.spaces[key]
    assert isinstance(space.space, DictSpace)
    value: Dict[str, int] = env.observation[key]
    assert len(value) == 56

    assert space.deterministic
    assert not space.platform_dependent


def test_programl_observation_space(env: LlvmEnv):
    env.reset("cBench-v0/crc32")
    key = "Programl"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Sequence)
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

    assert space.deterministic
    assert not space.platform_dependent


def test_cpuinfo_observation_space(env: LlvmEnv):
    env.reset("cBench-v0/crc32")
    key = "CpuInfo"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Sequence)
    value: Dict[str, Any] = env.observation[key]
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

    assert space.deterministic
    assert space.platform_dependent

    key = "CpuInfoDict"
    space = env.observation.spaces[key]
    value: Dict[str, Any] = env.observation[key]
    assert space.deterministic
    assert space.platform_dependent


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
    env: LlvmEnv, cbench_crc32_inst2vec_embedding_indices: List[int]
):
    env.reset("cBench-v0/crc32")
    key = "Inst2vecPreprocessedText"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Sequence)
    value: List[str] = env.observation[key]

    assert isinstance(value, list)
    for item, idx in zip(value, cbench_crc32_inst2vec_embedding_indices):
        assert isinstance(item, str)
    unk = env.observation.inst2vec.vocab["!UNK"]
    indices = [env.observation.inst2vec.vocab.get(item, unk) for item in value]
    assert indices == cbench_crc32_inst2vec_embedding_indices

    assert space.deterministic
    assert not space.platform_dependent


def test_inst2vec_embedding_indices_observation_space(
    env: LlvmEnv, cbench_crc32_inst2vec_embedding_indices: List[int]
):
    env.reset("cBench-v0/crc32")
    key = "Inst2vecEmbeddingIndices"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Sequence)
    value: List[int] = env.observation[key]

    print(value)
    assert isinstance(value, list)
    for item in value:
        assert isinstance(item, int)
    assert value == cbench_crc32_inst2vec_embedding_indices

    assert space.deterministic
    assert not space.platform_dependent


def test_inst2vec_observation_space(
    env: LlvmEnv, cbench_crc32_inst2vec_embedding_indices: List[int]
):
    env.reset("cBench-v0/crc32")
    key = "Inst2vec"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Sequence)
    value: np.ndarray = env.observation[key]

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

    assert space.deterministic
    assert not space.platform_dependent


def test_ir_instruction_count_observation_spaces(env: LlvmEnv):
    env.reset("cBench-v0/crc32")

    key = "IrInstructionCount"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Box)
    assert space.deterministic
    assert not space.platform_dependent
    value: np.ndarray = env.observation[key]
    assert isinstance(value, np.ndarray)
    np.testing.assert_array_equal([196], value)

    key = "IrInstructionCountO0"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Box)
    assert space.deterministic
    assert not space.platform_dependent
    value: np.ndarray = env.observation[key]
    assert isinstance(value, np.ndarray)
    np.testing.assert_array_equal([196], value)

    key = "IrInstructionCountO3"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Box)
    assert space.deterministic
    assert not space.platform_dependent
    value: np.ndarray = env.observation[key]
    assert isinstance(value, np.ndarray)
    np.testing.assert_array_equal([125], value)

    key = "IrInstructionCountOz"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Box)
    assert space.deterministic
    assert not space.platform_dependent
    value: np.ndarray = env.observation[key]
    assert isinstance(value, np.ndarray)
    np.testing.assert_array_equal([105], value)


def test_object_text_size_observation_spaces(env: LlvmEnv):
    env.reset("cBench-v0/crc32")

    # Expected .text sizes for this benchmark: -O0, -O3, -Oz.
    crc32_code_sizes = {"darwin": [1141, 3502, 3265], "linux": [1111, 3480, 3251]}
    print("ObjectTextSizeO0", env.observation["ObjectTextSizeO0"])
    print("ObjectTextSizeO3", env.observation["ObjectTextSizeO3"])
    print("ObjectTextSizeOz", env.observation["ObjectTextSizeOz"])

    key = "ObjectTextSizeBytes"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Box)
    assert space.deterministic
    assert space.platform_dependent
    value: np.ndarray = env.observation[key]
    assert isinstance(value, np.ndarray)
    np.testing.assert_array_equal(crc32_code_sizes[sys.platform][0], value)

    key = "ObjectTextSizeO0"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Box)
    assert space.deterministic
    assert space.platform_dependent
    value: np.ndarray = env.observation[key]
    assert isinstance(value, np.ndarray)
    np.testing.assert_array_equal(crc32_code_sizes[sys.platform][0], value)

    key = "ObjectTextSizeO3"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Box)
    assert space.deterministic
    assert space.platform_dependent
    value: np.ndarray = env.observation[key]
    assert isinstance(value, np.ndarray)
    np.testing.assert_array_equal(crc32_code_sizes[sys.platform][1], value)

    key = "ObjectTextSizeOz"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Box)
    assert space.deterministic
    assert space.platform_dependent
    value: np.ndarray = env.observation[key]
    assert isinstance(value, np.ndarray)
    np.testing.assert_array_equal(crc32_code_sizes[sys.platform][2], value)


if __name__ == "__main__":
    main()
