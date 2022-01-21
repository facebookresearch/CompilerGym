# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_test(
  NAME
    llvm_instcount_test
  SRCS
    "llvm_instcount_test.py"
  DEPS
    compiler_gym::leaderboard::llvm_instcount
    tests::pytest_plugins::common
    tests::test_main
)
