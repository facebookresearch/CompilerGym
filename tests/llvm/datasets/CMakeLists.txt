# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_test(
  NAME
    anghabench_test
  SRCS
    "anghabench_test.py"
  DEPS
    compiler_gym::envs::llvm::llvm
    compiler_gym::envs::llvm::datasets::datasets
    tests::pytest_plugins::common
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_py_test(
  NAME
    jotaibench_test
  SRCS
    "jotaibench_test.py"
  DEPS
    compiler_gym::envs::llvm::llvm
    compiler_gym::envs::llvm::datasets::datasets
    tests::pytest_plugins::common
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_py_test(
  NAME
    cbench_test
  SRCS
    "cbench_test.py"
  DEPS
    compiler_gym::envs::llvm::llvm
    compiler_gym::envs::llvm::datasets::datasets
    tests::pytest_plugins::common
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_py_test(
  NAME
    cbench_validate_test
  SRCS
    "cbench_validate_test.py"
  DEPS
    compiler_gym::compiler_gym
    compiler_gym::datasets::datasets
    compiler_gym::envs::envs
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_py_test(
  NAME
    chstone_test
  SRCS
    "chstone_test.py"
  DEPS
    compiler_gym::envs::llvm::llvm
    compiler_gym::envs::llvm::datasets::datasets
    tests::pytest_plugins::common
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_py_test(
  NAME
    clgen_test
  SRCS
    "clgen_test.py"
  DEPS
    compiler_gym::envs::llvm::llvm
    compiler_gym::envs::llvm::datasets::datasets
    tests::pytest_plugins::common
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_py_test(
  NAME
    csmith_test
  SRCS
    "csmith_test.py"
  DEPS
    compiler_gym::envs::llvm::llvm
    compiler_gym::envs::llvm::datasets::datasets
    compiler_gym::errors::errors
    tests::pytest_plugins::common
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_py_test(
  NAME
    github_test
  SRCS
    "github_test.py"
  DEPS
    compiler_gym::envs::llvm::llvm
    compiler_gym::envs::llvm::datasets::datasets
    tests::pytest_plugins::common
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_py_test(
  NAME
    llvm_datasets_test
  SRCS
    "llvm_datasets_test.py"
  DEPS
    compiler_gym::envs::llvm::llvm
    tests::test_main
)

cg_py_test(
  NAME
    llvm_stress_test
  SRCS
    "llvm_stress_test.py"
  DEPS
    compiler_gym::envs::llvm::llvm
    compiler_gym::envs::llvm::datasets::datasets
    compiler_gym::errors::errors
    tests::pytest_plugins::common
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_py_test(
  NAME
    poj104_test
  SRCS
    "poj104_test.py"
  DEPS
    compiler_gym::envs::llvm::llvm
    compiler_gym::envs::llvm::datasets::datasets
    tests::pytest_plugins::common
    tests::pytest_plugins::llvm
    tests::test_main
)
