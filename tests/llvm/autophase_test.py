# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the LLVM CompilerGym environments."""
from compiler_gym.envs import CompilerEnv
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_autophase_crc32_feature_vector(env: CompilerEnv):
    env.reset(benchmark="cbench-v1/crc32")
    print(env.benchmark)  # For debugging in case of error.
    features = env.observation["AutophaseDict"]
    print(features)  # For debugging on failure.
    assert features == {
        "BBNumArgsHi": 0,
        "BBNumArgsLo": 0,
        "onePred": 16,
        "onePredOneSuc": 12,
        "onePredTwoSuc": 2,
        "oneSuccessor": 16,
        "twoPred": 8,
        "twoPredOneSuc": 2,
        "twoEach": 4,
        "twoSuccessor": 8,
        "morePreds": 0,
        "BB03Phi": 0,
        "BBHiPhi": 0,
        "BBNoPhi": 29,
        "BeginPhi": 0,
        "BranchCount": 24,
        "returnInt": 9,
        "CriticalCount": 2,
        "NumEdges": 32,
        "const32Bit": 44,
        "const64Bit": 41,
        "numConstZeroes": 14,
        "numConstOnes": 36,
        "UncondBranches": 16,
        "binaryConstArg": 13,
        "NumAShrInst": 0,
        "NumAddInst": 5,
        "NumAllocaInst": 26,
        "NumAndInst": 3,
        "BlockMid": 5,
        "BlockLow": 24,
        "NumBitCastInst": 20,
        "NumBrInst": 24,
        "NumCallInst": 33,
        "NumGetElementPtrInst": 5,
        "NumICmpInst": 10,
        "NumLShrInst": 3,
        "NumLoadInst": 51,
        "NumMulInst": 0,
        "NumOrInst": 1,
        "NumPHIInst": 0,
        "NumRetInst": 5,
        "NumSExtInst": 0,
        "NumSelectInst": 0,
        "NumShlInst": 0,
        "NumStoreInst": 42,
        "NumSubInst": 0,
        "NumTruncInst": 1,
        "NumXorInst": 8,
        "NumZExtInst": 5,
        "TotalBlocks": 29,
        "TotalInsts": 242,
        "TotalMemInst": 157,
        "TotalFuncs": 15,
        "ArgsPhi": 0,
        "testUnary": 103,
    }


if __name__ == "__main__":
    main()
