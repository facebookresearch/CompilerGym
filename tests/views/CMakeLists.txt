# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_test(
  NAME
    observation_test
  SRCS
    "observation_test.py"
  DEPS
    compiler_gym::service::proto::proto
    compiler_gym::views::views
    tests::test_main
)

cg_py_test(
  NAME
    reward_test
  SRCS
    "reward_test.py"
  DEPS
    compiler_gym::views::views
    tests::test_main
)
