# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

py_test(
  NAME py_converters_test
  SRCS "py_converters_test.py"
  DEPS
    compiler_gym::service::proto::proto
    compiler_gym::spaces::spaces
    tests::test_main
)
