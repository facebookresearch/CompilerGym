# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_test(
  NAME "gccinvocation_test"
  SRCS "gccinvocation_test.py"
  DEPS
    compiler_gym::third_party::gccinvocation
    tests::test_main
)
