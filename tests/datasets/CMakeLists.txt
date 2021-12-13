# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_test(
  NAME
    benchmark_test
  SRCS
    "benchmark_test.py"
  DEPS
    compiler_gym::datasets::datasets
    tests::pytest_plugins::common
    tests::test_main
)

cg_py_test(
  NAME
    dataset_test
  SRCS
    "dataset_test.py"
  DEPS
    compiler_gym::datasets::datasets
    tests::pytest_plugins::common
    tests::test_main
)

cg_py_test(
  NAME
    datasets_test
  SRCS
    "datasets_test.py"
  DEPS
    compiler_gym::datasets::datasets
    tests::pytest_plugins::common
    tests::test_main
)

cg_py_test(
  NAME
    files_dataset_test
  SRCS
    "files_dataset_test.py"
  DEPS
    compiler_gym::datasets::datasets
    tests::pytest_plugins::common
    tests::test_main
)
