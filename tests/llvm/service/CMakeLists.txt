# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_cc_test(
  NAME
    ActionSpaceTest
  SRCS
    "ActionSpaceTest.cc"
  DEPS
    compiler_gym::envs::llvm::service::ActionSpace
    tests::TestMacros
    tests::TestMain
  ABS_DEPS
    GTest::GTest
    magic_enum
)
