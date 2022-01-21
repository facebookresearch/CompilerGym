# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_test(
  NAME "actions_test"
  SRCS "actions_test.py"
  DEPS
    compiler_gym::compiler_gym
    tests::test_main
)
