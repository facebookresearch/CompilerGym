# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the LLVM CompilerGym environments."""
import os
import sys
from typing import Any, Dict, List, NamedTuple

import gym
import networkx as nx
import numpy as np
import pytest
from flaky import flaky

from compiler_gym.envs.llvm.llvm_env import LlvmEnv
from compiler_gym.spaces import Box
from compiler_gym.spaces import Dict as DictSpace
from compiler_gym.spaces import Scalar, Sequence
from tests.pytest_plugins.common import ci_only
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_default_observation_space(env: LlvmEnv):
    env.observation_space = "Autophase"
    assert env.observation_space.shape == (56,)
    assert env.observation_space_spec.id == "Autophase"

    env.observation_space = None
    assert env.observation_space is None
    assert env.observation_space_spec is None

    invalid = "invalid value"
    with pytest.raises(LookupError, match=f"Observation space not found: {invalid}"):
        env.observation_space = invalid


def test_observation_spaces(env: LlvmEnv):
    env.reset("cbench-v1/crc32")

    assert set(env.observation.spaces.keys()) == {
        "Autophase",
        "AutophaseDict",
        "Bitcode",
        "BitcodeFile",
        "Buildtime",
        "CpuInfo",
        "Inst2vec",
        "Inst2vecEmbeddingIndices",
        "Inst2vecPreprocessedText",
        "InstCount",
        "InstCountDict",
        "InstCountNorm",
        "InstCountNormDict",
        "Ir",
        "Ir2vecFlowAware",
        "Ir2vecSymbolic",
        "Ir2vecFunctionLevelFlowAware",
        "Ir2vecFunctionLevelSymbolic",
        "IrInstructionCount",
        "IrInstructionCountO0",
        "IrInstructionCountO3",
        "IrInstructionCountOz",
        "IrSha1",
        "IsBuildable",
        "IsRunnable",
        "LexedIr",
        "LexedIrTuple",
        "ObjectTextSizeBytes",
        "ObjectTextSizeO0",
        "ObjectTextSizeO3",
        "ObjectTextSizeOz",
        "Programl",
        "ProgramlJson",
        "Runtime",
        "TextSizeBytes",
        "TextSizeO0",
        "TextSizeO3",
        "TextSizeOz",
    }


def test_ir_observation_space(env: LlvmEnv):
    env.reset("cbench-v1/crc32")
    key = "Ir"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Sequence)
    assert space.space.dtype == str
    assert space.space.size_range == (0, np.iinfo(np.int64).max)

    value: str = env.observation[key]
    print(value)  # For debugging in case of error.
    assert isinstance(value, str)
    assert space.space.contains(value)

    assert space.deterministic
    assert not space.platform_dependent


def test_ir_sha1_observation_space(env: LlvmEnv):
    env.reset("cbench-v1/crc32")
    key = "IrSha1"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Sequence)
    assert space.space.dtype == str
    assert space.space.size_range == (40, 40)

    value: str = env.observation[key]
    print(value)  # For debugging in case of error.
    assert isinstance(value, str)
    assert len(value) == 40
    assert space.space.contains(value)

    assert space.deterministic
    assert not space.platform_dependent


def test_bitcode_observation_space(env: LlvmEnv):
    env.reset("cbench-v1/crc32")
    key = "Bitcode"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Sequence)
    assert space.space.dtype == np.int8
    assert space.space.size_range == (0, np.iinfo(np.int64).max)

    assert space.deterministic
    assert not space.platform_dependent

    value: str = env.observation[key]
    print(value)  # For debugging in case of error.
    assert isinstance(value, np.ndarray)
    assert value.dtype == np.int8
    assert space.space.contains(value)


def test_bitcode_file_observation_space(env: LlvmEnv):
    env.reset("cbench-v1/crc32")
    key = "BitcodeFile"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Sequence)
    assert space.space.dtype == str
    assert space.space.size_range == (0, 4096)
    assert not space.deterministic
    assert not space.platform_dependent

    value: str = env.observation[key]
    print(value)  # For debugging in case of error.
    try:
        assert isinstance(value, str)
        assert os.path.isfile(value)
        assert space.space.contains(value)
    finally:
        os.unlink(value)


@pytest.mark.parametrize(
    "benchmark_uri", ["cbench-v1/crc32", "cbench-v1/qsort", "cbench-v1/gsm"]
)
def test_bitcode_file_equivalence(env: LlvmEnv, benchmark_uri: str):
    """Test that LLVM produces the same bitcode as a file and as a byte array."""
    env.reset(benchmark=benchmark_uri)

    bitcode = env.observation.Bitcode()
    bitcode_file = env.observation.BitcodeFile()

    try:
        with open(bitcode_file, "rb") as f:
            bitcode_from_file = f.read()

        assert bitcode.tobytes() == bitcode_from_file
    finally:
        os.unlink(bitcode_file)


# The Autophase feature vector for benchmark://cbench-v1/crc32 in its initial
# state.
AUTOPHASE_CBENCH_CRC32 = [
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
    44,
    41,
    14,
    36,
    16,
    13,
    0,
    5,
    26,
    3,
    5,
    24,
    20,
    24,
    33,
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
    42,
    0,
    1,
    8,
    5,
    29,
    242,
    157,
    15,
    0,
    103,
]


def test_autophase_observation_space_reset(env: LlvmEnv):
    """Test that the intial observation is returned on env.reset()."""
    env.observation_space = "Autophase"
    observation = env.reset("cbench-v1/crc32")
    print(observation.tolist())  # For debugging on error.
    np.testing.assert_array_equal(observation, AUTOPHASE_CBENCH_CRC32)


def test_instcount_observation_space(env: LlvmEnv):
    env.reset("cbench-v1/crc32")
    key = "InstCount"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Box)
    assert space.space.dtype == np.int64
    assert space.space.shape == (70,)
    assert space.deterministic
    assert not space.platform_dependent

    value: np.ndarray = env.observation[key]
    print(value.tolist())  # For debugging in case of error.

    expected_values = [
        242,
        29,
        15,
        5,
        24,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        5,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        3,
        0,
        3,
        1,
        8,
        26,
        51,
        42,
        5,
        0,
        0,
        0,
        1,
        5,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        20,
        0,
        0,
        0,
        10,
        0,
        0,
        33,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]

    np.testing.assert_array_equal(value, expected_values)
    assert value.dtype == np.int64

    # The first value is the total number of instructions. This should equal the
    # number of instructions.
    assert sum(value[3:]) == value[0]


def test_instcount_dict_observation_space(env: LlvmEnv):
    env.reset("cbench-v1/crc32")
    key = "InstCountDict"
    space = env.observation.spaces[key]
    assert isinstance(space.space, DictSpace)
    assert space.deterministic
    assert not space.platform_dependent

    value: Dict[str, int] = env.observation[key]
    print(value)  # For debugging in case of error.
    assert len(value) == 70


def test_instcount_norm_observation_space(env: LlvmEnv):
    env.reset("cbench-v1/crc32")
    key = "InstCountNorm"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Box)
    assert space.space.dtype == np.float32
    assert space.space.shape == (69,)
    assert space.deterministic
    assert not space.platform_dependent

    value: np.ndarray = env.observation[key]
    print(value.tolist())  # For debugging in case of error.

    assert value.shape == (69,)
    assert value.dtype == np.float32

    # Assert that the normalized instruction counts sum to 1. Note that the
    # first two features (#blocks and #funcs) must be excluded.
    assert pytest.approx(sum(value[2:]), 1.0)


def test_instcount_norm_dict_observation_space(env: LlvmEnv):
    env.reset("cbench-v1/crc32")
    key = "InstCountNormDict"
    space = env.observation.spaces[key]
    assert isinstance(space.space, DictSpace)
    assert space.deterministic
    assert not space.platform_dependent

    value: Dict[str, int] = env.observation[key]
    print(value)  # For debugging in case of error.
    assert len(value) == 69


def test_autophase_observation_space(env: LlvmEnv):
    env.reset("cbench-v1/crc32")
    key = "Autophase"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Box)

    value: np.ndarray = env.observation[key]
    print(value.tolist())  # For debugging in case of error.
    assert isinstance(value, np.ndarray)
    assert value.shape == (56,)

    assert space.deterministic
    assert not space.platform_dependent

    np.testing.assert_array_equal(value, AUTOPHASE_CBENCH_CRC32)
    assert space.space.contains(value)


def test_autophase_dict_observation_space(env: LlvmEnv):
    env.reset("cbench-v1/crc32")
    key = "AutophaseDict"
    space = env.observation.spaces[key]
    assert isinstance(space.space, DictSpace)
    value: Dict[str, int] = env.observation[key]
    print(value)  # For debugging in case of error.
    assert len(value) == 56

    assert space.deterministic
    assert not space.platform_dependent


def test_lexed_ir_observation_space(env: LlvmEnv):
    env.reset("cbench-v1/crc32")
    key = "LexedIr"
    space = env.observation.spaces[key]
    print(type(space.space))
    print(space.space)
    assert isinstance(space.space, Sequence)
    value: Dict[str, np.array] = env.observation[key]
    print(value)  # For debugging in case of error
    assert len(value) == 4

    assert space.deterministic
    assert not space.platform_dependent


def test_lexed_ir_tuple_observation_space(env: LlvmEnv):
    env.reset("cbench-v1/crc32")
    key = "LexedIrTuple"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Sequence)
    value: List[NamedTuple] = env.observation[key]
    print(value)  # For debugging in case of error

    assert space.deterministic
    assert not space.platform_dependent


def test_programl_observation_space(env: LlvmEnv):
    env.reset("cbench-v1/crc32")
    key = "Programl"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Sequence)
    graph: nx.MultiDiGraph = env.observation[key]
    assert isinstance(graph, nx.MultiDiGraph)

    assert graph.number_of_nodes() == 512
    assert graph.number_of_edges() == 907
    assert graph.nodes[0] == {
        "block": 0,
        "function": 0,
        "text": "[external]",
        "type": 0,
    }

    assert space.deterministic
    assert not space.platform_dependent


def test_programl_json_observation_space(env: LlvmEnv):
    env.reset("cbench-v1/crc32")
    key = "ProgramlJson"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Sequence)
    graph: Dict[str, Any] = env.observation[key]
    assert isinstance(graph, dict)


def test_cpuinfo_observation_space(env: LlvmEnv):
    env.reset("cbench-v1/crc32")
    key = "CpuInfo"
    space = env.observation.spaces[key]
    assert isinstance(space.space, DictSpace)
    value: Dict[str, Any] = env.observation[key]
    print(value)  # For debugging in case of error.
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


@pytest.fixture
def cbench_crc32_inst2vec_embedding_indices() -> List[int]:
    """The expected inst2vec embedding indices for cbench-v1/crc32."""
    # The linux/macOS builds of clang produce slightly different bitcodes.
    if sys.platform.lower().startswith("linux"):
        return [
            8564,
            8564,
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
            289,
            200,
            1412,
            1412,
            8564,
            3032,
            180,
            3032,
            293,
            3032,
            205,
            415,
            205,
            213,
            8564,
            8564,
            8564,
            204,
            8564,
            213,
            215,
            364,
            364,
            216,
            8564,
            216,
            8564,
            8564,
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
            8564,
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
            8564,
            8564,
            182,
            961,
            214,
            415,
            214,
            364,
            364,
            216,
            8564,
            293,
            3032,
            180,
            3032,
            8564,
            3032,
            295,
            257,
            8564,
            291,
            178,
            178,
            200,
            214,
            180,
            3032,
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
            180,
            3032,
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
            180,
            3032,
            180,
            3032,
            293,
            3032,
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
            293,
            3032,
            180,
            3032,
            180,
            3032,
            257,
            8564,
            289,
            289,
            8564,
            8564,
            178,
            178,
            289,
            364,
            311,
            594,
            8564,
            3032,
            8564,
            180,
            3032,
            180,
            3032,
            8564,
            8564,
            8564,
            204,
            8564,
            8564,
            8564,
            364,
            364,
            216,
            8564,
            8564,
            8564,
            8564,
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
            364,
            216,
            8564,
            180,
            3032,
            180,
            3032,
            8564,
            3032,
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
            289,
            200,
            1412,
            1412,
            8564,
            3032,
            180,
            3032,
            293,
            3032,
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
            364,
            216,
            8564,
            293,
            3032,
            180,
            3032,
            8564,
            3032,
            295,
            257,
            8564,
            291,
            178,
            178,
            200,
            214,
            180,
            3032,
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
            180,
            3032,
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
            180,
            3032,
            180,
            3032,
            293,
            3032,
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
            293,
            3032,
            180,
            3032,
            180,
            3032,
            257,
            8564,
            289,
            289,
            8564,
            8564,
            178,
            178,
            289,
            364,
            311,
            594,
            8564,
            3032,
            8564,
            180,
            3032,
            180,
            3032,
            8564,
            8564,
            5666,
            204,
            8564,
            5391,
            8564,
            364,
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
            364,
            216,
            8564,
            180,
            3032,
            180,
            3032,
            8564,
            3032,
            295,
            257,
        ]
    else:
        raise NotImplementedError(f"Unknown platform: {sys.platform}")


def test_inst2vec_preprocessed_observation_space(
    env: LlvmEnv, cbench_crc32_inst2vec_embedding_indices: List[int]
):
    env.reset("cbench-v1/crc32")
    key = "Inst2vecPreprocessedText"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Sequence)
    value: List[str] = env.observation[key]

    assert isinstance(value, list)
    for item, idx in zip(value, cbench_crc32_inst2vec_embedding_indices):
        assert isinstance(item, str)
    unk = env.inst2vec.vocab["!UNK"]
    indices = [env.inst2vec.vocab.get(item, unk) for item in value]
    print(indices)  # For debugging in case of error.
    assert indices == cbench_crc32_inst2vec_embedding_indices

    assert space.deterministic
    assert not space.platform_dependent


def test_inst2vec_embedding_indices_observation_space(
    env: LlvmEnv, cbench_crc32_inst2vec_embedding_indices: List[int]
):
    env.reset("cbench-v1/crc32")
    key = "Inst2vecEmbeddingIndices"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Sequence)
    value: List[int] = env.observation[key]
    print(value)  # For debugging in case of error.

    assert isinstance(value, list)
    for item in value:
        assert isinstance(item, int)
    assert value == cbench_crc32_inst2vec_embedding_indices

    assert space.deterministic
    assert not space.platform_dependent


def test_inst2vec_observation_space(
    env: LlvmEnv, cbench_crc32_inst2vec_embedding_indices: List[int]
):
    env.reset("cbench-v1/crc32")
    key = "Inst2vec"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Sequence)
    value: np.ndarray = env.observation[key]
    print(value)  # For debugging in case of error.

    assert isinstance(value, np.ndarray)
    assert value.dtype == np.float32
    height, width = value.shape
    assert width == len(env.inst2vec.embeddings[0])
    assert height == len(cbench_crc32_inst2vec_embedding_indices)
    # Check a handful of values.
    np.testing.assert_array_almost_equal(
        value.tolist(),
        [
            env.inst2vec.embeddings[idx]
            for idx in cbench_crc32_inst2vec_embedding_indices
        ],
    )

    assert space.deterministic
    assert not space.platform_dependent


def test_ir_instruction_count_observation_spaces(env: LlvmEnv):
    env.reset("cbench-v1/crc32")

    key = "IrInstructionCount"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Scalar)
    assert space.deterministic
    assert not space.platform_dependent
    value: int = env.observation[key]
    print(value)  # For debugging in case of error.
    assert isinstance(value, int)
    assert value == 242

    key = "IrInstructionCountO0"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Scalar)
    assert space.deterministic
    assert not space.platform_dependent
    value: int = env.observation[key]
    print(value)  # For debugging in case of error.
    assert isinstance(value, int)
    assert value == 242

    key = "IrInstructionCountO3"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Scalar)
    assert space.deterministic
    assert not space.platform_dependent
    value: int = env.observation[key]
    print(value)  # For debugging in case of error.
    assert isinstance(value, int)
    assert value == 164

    key = "IrInstructionCountOz"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Scalar)
    assert space.deterministic
    assert not space.platform_dependent
    value: int = env.observation[key]
    print(value)  # For debugging in case of error.
    assert isinstance(value, int)
    assert value == 114


def test_object_text_size_observation_spaces(env: LlvmEnv):
    env.reset("cbench-v1/crc32")

    # Expected .text sizes for this benchmark: -O0, -O3, -Oz.
    crc32_code_sizes = {"darwin": [1171, 3825, 3289], "linux": [1183, 3961, 3286]}
    actual_code_sizes = []

    key = "ObjectTextSizeBytes"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Scalar)
    assert space.deterministic
    assert space.platform_dependent
    value: int = env.observation[key]
    assert isinstance(value, int)
    actual_code_sizes.append(value)

    key = "ObjectTextSizeO0"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Scalar)
    assert space.deterministic
    assert space.platform_dependent
    value: int = env.observation[key]
    assert isinstance(value, int)

    key = "ObjectTextSizeO3"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Scalar)
    assert space.deterministic
    assert space.platform_dependent
    value: int = env.observation[key]
    assert isinstance(value, int)
    actual_code_sizes.append(value)

    key = "ObjectTextSizeOz"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Scalar)
    assert space.deterministic
    assert space.platform_dependent
    value: int = env.observation[key]
    assert isinstance(value, int)
    actual_code_sizes.append(value)

    # For debugging in case of error:
    print("Expected code sizes:", crc32_code_sizes[sys.platform])
    print("Actual code sizes:", actual_code_sizes)
    assert crc32_code_sizes[sys.platform] == actual_code_sizes


def test_text_size_observation_spaces(env: LlvmEnv):
    env.reset("cbench-v1/crc32")

    key = "TextSizeBytes"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Scalar)
    assert space.deterministic
    assert space.platform_dependent
    value: int = env.observation[key]
    print(value)  # For debugging in case of error.
    assert isinstance(value, int)

    key = "TextSizeO0"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Scalar)
    assert space.deterministic
    assert space.platform_dependent
    value: int = env.observation[key]
    print(value)  # For debugging in case of error.
    assert isinstance(value, int)
    assert value > 0  # Exact value is system dependent, see below.

    key = "TextSizeO3"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Scalar)
    assert space.deterministic
    assert space.platform_dependent
    value: int = env.observation[key]
    print(value)  # For debugging in case of error.
    assert isinstance(value, int)
    assert value > 0  # Exact value is system dependent, see below.

    key = "TextSizeOz"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Scalar)
    assert space.deterministic
    assert space.platform_dependent
    value: int = env.observation[key]
    print(value)  # For debugging in case of error.
    assert isinstance(value, int)
    assert value > 0  # Exact value is system dependent, see below.


# NOTE(cummins): The exact values here depend on the system toolchain and
# libraries, so only run this test on the GitHub CI runner environment where we
# can hardcode the values. If this test starts to fail, it may be because the CI
# runner environment has changed.
@ci_only
def test_text_size_observation_space_values(env: LlvmEnv):
    env.reset("cbench-v1/crc32")

    # Expected .text sizes for this benchmark: -O0, -O3, -Oz.
    crc32_code_sizes = {"darwin": [16384, 16384, 16384], "linux": [2850, 5652, 4980]}

    # For debugging in case of error.
    print(env.observation["TextSizeO0"])
    print(env.observation["TextSizeO3"])
    print(env.observation["TextSizeOz"])

    assert env.observation.TextSizeO0() == crc32_code_sizes[sys.platform][0]
    assert env.observation.TextSizeO0() == crc32_code_sizes[sys.platform][0]
    assert env.observation.TextSizeO3() == crc32_code_sizes[sys.platform][1]
    assert env.observation.TextSizeOz() == crc32_code_sizes[sys.platform][2]


@flaky  # Runtimes can timeout
def test_runtime_observation_space(env: LlvmEnv):
    env.reset("cbench-v1/crc32")
    key = "Runtime"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Sequence)

    value: np.ndarray = env.observation[key]
    print(value.tolist())  # For debugging in case of error.
    assert isinstance(value, np.ndarray)
    assert env.runtime_observation_count == 1
    assert value.shape == (1,)

    assert not space.deterministic
    assert space.platform_dependent

    assert space.space.contains(value)

    for buildtime in value:
        assert buildtime > 0


@flaky  # Runtimes can timeout
def test_runtime_observation_space_different_observation_count(env: LlvmEnv):
    """Test setting a custom observation count for LLVM runtimes."""
    env.reset("cbench-v1/crc32")

    env.runtime_observation_count = 3
    value: np.ndarray = env.observation["Runtime"]
    print(value.tolist())  # For debugging in case of error.
    assert value.shape == (3,)

    env.reset()
    value: np.ndarray = env.observation["Runtime"]
    print(value.tolist())  # For debugging in case of error.
    assert value.shape == (3,)

    env.runtime_observation_count = 5
    value: np.ndarray = env.observation["Runtime"]
    print(value.tolist())  # For debugging in case of error.
    assert value.shape == (5,)


@flaky  # Runtimes can timeout
def test_runtime_observation_space_invalid_observation_count(env: LlvmEnv):
    """Test setting an invalid custom observation count for LLVM runtimes."""
    env.reset("cbench-v1/crc32")

    val = env.runtime_observation_count
    with pytest.raises(
        ValueError, match="runtimes_per_observation_count must be >= 1. Received: -5"
    ):
        env.runtime_observation_count = -5
    assert env.runtime_observation_count == val  # unchanged


def test_runtime_observation_space_not_runnable(env: LlvmEnv):
    env.reset("chstone-v0/gsm")
    key = "Runtime"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Sequence)
    assert env.observation[key] is None


@flaky  # Build can timeout
def test_buildtime_observation_space(env: LlvmEnv):
    env.reset("cbench-v1/crc32")
    key = "Buildtime"

    space = env.observation.spaces[key]
    assert isinstance(space.space, Sequence)
    assert not space.deterministic
    assert space.platform_dependent

    value: np.ndarray = env.observation[key]
    print(value)  # For debugging in case of error.
    assert value.shape == (1,)
    assert space.space.contains(value)
    assert value[0] >= 0


def test_buildtime_observation_space_not_runnable(env: LlvmEnv):
    env.reset("chstone-v0/gsm")
    key = "Buildtime"

    space = env.observation.spaces[key]
    assert isinstance(space.space, Sequence)
    assert not space.deterministic
    assert space.platform_dependent

    assert env.observation[key] is None


def test_is_runnable_observation_space(env: LlvmEnv):
    env.reset("cbench-v1/crc32")
    key = "IsRunnable"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Scalar)
    assert space.deterministic
    assert space.platform_dependent
    value: int = env.observation[key]
    print(value)  # For debugging in case of error.
    assert isinstance(value, int)
    assert value == 1


def test_is_runnable_observation_space_not_runnable(env: LlvmEnv):
    env.reset("chstone-v0/gsm")
    key = "IsRunnable"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Scalar)
    assert space.deterministic
    assert space.platform_dependent
    value: int = env.observation[key]
    print(value)  # For debugging in case of error.
    assert isinstance(value, int)
    assert value == 0


def test_is_buildable_observation_space(env: LlvmEnv):
    env.reset("cbench-v1/crc32")
    key = "IsBuildable"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Scalar)
    assert space.deterministic
    assert space.platform_dependent
    value: int = env.observation[key]
    print(value)  # For debugging in case of error.
    assert isinstance(value, int)
    assert value == 1


def test_is_buildable_observation_space_not_buildable(env: LlvmEnv):
    env.reset("chstone-v0/gsm")
    key = "IsBuildable"
    space = env.observation.spaces[key]
    assert isinstance(space.space, Scalar)
    assert space.deterministic
    assert space.platform_dependent
    value: int = env.observation[key]
    print(value)  # For debugging in case of error.
    assert isinstance(value, int)
    assert value == 0


@pytest.mark.parametrize("name", ["Ir2vecFlowAware", "Ir2vecSymbolic"])
def test_ir2vec(env: LlvmEnv, name: str):
    env.reset()
    space = env.observation.spaces[name]
    assert isinstance(space.space, Box)
    value: np.ndarray = env.observation[name]

    assert space.space.dtype == np.float32
    assert space.space.shape == (300,)
    assert space.deterministic
    assert not space.platform_dependent

    np.testing.assert_array_almost_equal(
        space.space.low, np.full((300,), np.finfo(np.float32).min)
    )
    np.testing.assert_array_almost_equal(
        space.space.high, np.full((300,), np.finfo(np.float32).max)
    )

    assert isinstance(value, np.ndarray)
    assert value.shape == (300,)
    assert space.space.contains(value)


@pytest.mark.parametrize(
    "name", ["Ir2vecFunctionLevelFlowAware", "Ir2vecFunctionLevelSymbolic"]
)
def test_ir2vec_function_level(env: LlvmEnv, name: str):
    env.reset()
    space = env.observation.spaces[name]
    assert isinstance(space.space, Sequence)
    value: Dict[str, List[float]] = env.observation[name]

    assert value
    for k, v in value.items():
        assert isinstance(k, str)
        assert isinstance(v, list)
        assert len(v) == 300


@pytest.mark.xfail(
    reason="TODO(cummins): contains() method is broken for opaque types", strict=True
)
@pytest.mark.parametrize(
    "name", ["Ir2vecFunctionLevelFlowAware", "Ir2vecFunctionLevelSymbolic"]
)
def test_ir2vec_function_level_(env: LlvmEnv, name: str):
    env.reset()
    space = env.observation.spaces[name]
    value: Dict[str, List[float]] = env.observation[name]
    assert space.space.contains(value)


def test_add_derived_space(env: LlvmEnv):
    env.reset()
    env.observation.add_derived_space(
        id="IrLen",
        base_id="Ir",
        space=Box(name="IrLen", low=0, high=float("inf"), shape=(1,), dtype=int),
        translate=lambda base: [15],
    )

    value = env.observation["IrLen"]
    assert isinstance(value, list)
    assert value == [15]

    # Repeat the above test using the generated bound method.
    value = env.observation.IrLen()
    assert isinstance(value, list)
    assert value == [15]


def test_derived_space_constructor():
    """Test that derived observation space can be specified at construction
    time.
    """
    with gym.make("llvm-v0") as env:
        env.observation_space = "AutophaseDict"
        a = env.reset()

    with gym.make("llvm-v0", observation_space="AutophaseDict") as env:
        b = env.reset()

    assert a == b


if __name__ == "__main__":
    main()
