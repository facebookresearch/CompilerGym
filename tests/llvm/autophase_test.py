# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the LLVM CompilerGym environments."""
from compiler_gym.envs import CompilerEnv
from tests.test_main import main

pytest_plugins = ["tests.llvm.fixtures"]


def test_autophase_crc32_feature_vector(env: CompilerEnv):
    env.benchmark = "cBench-v0/crc32"
    env.reset()
    features = env.observation["Autophase"]
    assert features[0] == 0  # BBNumArgsHi
    assert features[1] == 0  # BBNumArgsLo
    assert features[2] == 16  # onePred
    assert features[3] == 12  # onePredOneSuc
    assert features[4] == 2  # onePredTwoSuc
    assert features[5] == 16  # oneSuccessor
    assert features[6] == 8  # twoPred
    assert features[7] == 2  # twoPredOneSuc
    assert features[8] == 4  # twoEach
    assert features[9] == 8  # twoSuccessor
    assert features[10] == 0  # morePreds
    assert features[11] == 0  # BB03Phi
    assert features[12] == 0  # BBHiPhi
    assert features[13] == 29  # BBNoPhi
    assert features[14] == 0  # BeginPhi
    assert features[15] == 24  # BranchCount
    assert features[16] == 9  # returnInt
    assert features[17] == 2  # CriticalCount
    assert features[18] == 32  # NumEdges
    assert features[19] == 38  # const32Bit
    assert features[20] == 21  # const64Bit
    assert features[21] == 14  # numConstZeroes
    assert features[22] == 30  # numConstOnes
    assert features[23] == 16  # UncondBranches
    assert features[24] == 13  # binaryConstArg
    assert features[25] == 0  # NumAShrInst
    assert features[26] == 5  # NumAddInst
    assert features[27] == 24  # NumAllocaInst
    assert features[28] == 3  # NumAndInst
    assert features[29] == 3  # BlockMid
    assert features[30] == 26  # BlockLow
    assert features[31] == 0  # NumBitCastInst
    assert features[32] == 24  # NumBrInst
    assert features[33] == 13  # NumCallInst
    assert features[34] == 5  # NumGetElementPtrInst
    assert features[35] == 10  # NumICmpInst
    assert features[36] == 3  # NumLShrInst
    assert features[37] == 51  # NumLoadInst
    assert features[38] == 0  # NumMulInst
    assert features[39] == 1  # NumOrInst
    assert features[40] == 0  # NumPHIInst
    assert features[41] == 5  # NumRetInst
    assert features[42] == 0  # NumSExtInst
    assert features[43] == 0  # NumSelectInst
    assert features[44] == 0  # NumShlInst
    assert features[45] == 38  # NumStoreInst
    assert features[46] == 0  # NumSubInst
    assert features[47] == 1  # NumTruncInst
    assert features[48] == 8  # NumXorInst
    assert features[49] == 5  # NumZExtInst
    assert features[50] == 29  # TotalBlocks
    assert features[51] == 196  # TotalInsts
    assert features[52] == 131  # TotalMemInst
    assert features[53] == 13  # TotalFuncs
    assert features[54] == 0  # ArgsPhi
    assert features[55] == 81  # testUnary


if __name__ == "__main__":
    main()
