# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_test(
  NAME
    llvm_cbench_validate_fuzz_test
  SRCS
    "llvm_cbench_validate_fuzz_test.py"
  DEPS
    compiler_gym::compiler_gym
    tests::pytest_plugins::llvm
    tests::test_main
  LABELS
    "manual"
)

cg_py_test(
  NAME
    llvm_commandline_opt_equivalence_fuzz_test
  SRCS
    "llvm_commandline_opt_equivalence_fuzz_test.py"
  DEPS
    compiler_gym::compiler_gym
    tests::pytest_plugins::common
    tests::pytest_plugins::llvm
    tests::pytest_plugins::random_util
    tests::test_main
  LABELS
    "manual"
)

cg_py_test(
  NAME
    llvm_deterministic_action_fuzz_test
  SRCS
    "llvm_deterministic_action_fuzz_test.py"
  DEPS
    compiler_gym::compiler_gym
    tests::pytest_plugins::llvm
    tests::test_main
  LABELS
    "manual"
)

cg_py_test(
  NAME
    llvm_fork_env_fuzz_test
  SRCS
    "llvm_fork_env_fuzz_test.py"
  DEPS
    compiler_gym::compiler_gym
    tests::pytest_plugins::llvm
    tests::test_main
  LABELS
    "manual"
)

# TODO(boian): fix when benchmarks_random_actions_test.py is present in source.
#cg_py_test(
#  NAME
#    benchmarks_random_actions_test
#  SRCS
#    "benchmarks_random_actions_test.py"
#  DEPS
#    compiler_gym::compiler_gym
#    compiler_gym::envs::envs
#    compiler_gym::third_party::autophase::autophase
#    tests::pytest_plugins::llvm
#    tests::test_main
#  LABELS
#    "manual"
#)

cg_py_test(
  NAME
    llvm_trajectory_replay_fuzz_test
  SRCS
    "llvm_trajectory_replay_fuzz_test.py"
  DEPS
    compiler_gym::compiler_gym
    tests::pytest_plugins::llvm
    tests::pytest_plugins::random_util
    tests::test_main
  LABELS
    "manual"
)

cg_py_test(
  NAME
    llvm_stress_fuzz_test
  SRCS
    "llvm_stress_fuzz_test.py"
  DEPS
    compiler_gym::compiler_gym
    tests::pytest_plugins::llvm
    tests::pytest_plugins::random_util
    tests::test_main
  LABELS
    "manual"
)
