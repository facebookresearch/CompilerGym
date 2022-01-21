# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_test(
  NAME
    commandline_test
  SRCS
    "commandline_test.py"
  DEPS
    compiler_gym::spaces::spaces
    tests::test_main
)

cg_py_test(
  NAME
    named_discrete_test
  SRCS
    "named_discrete_test.py"
  DEPS
    compiler_gym::spaces::spaces
    tests::test_main
)

cg_py_test(
  NAME
    scalar_test
  SRCS
    "scalar_test.py"
  DEPS
    compiler_gym::spaces::spaces
    tests::test_main
)

cg_py_test(
  NAME
    sequence_test
  SRCS
    "sequence_test.py"
  DEPS
    compiler_gym::spaces::spaces
    tests::test_main
)
