# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_test(
  NAME action_space_test
  SRCS "action_space_test.py"
  DEPS
    compiler_gym::envs::envs
    tests::test_main
    tests::pytest_plugins::llvm
)

cg_py_test(
  NAME
    all_actions_single_step_test
  SRCS
    "all_actions_single_step_test.py"
  DEPS
    compiler_gym::envs::envs
    compiler_gym::third_party::autophase::autophase
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_py_test(
  NAME
    all_benchmarks_init_close_test
  SRCS
    "all_benchmarks_init_close_test.py"
  DEPS
    compiler_gym::envs::envs
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_py_test(
  NAME
    autophase_test
  SRCS
    "autophase_test.py"
  DEPS
    compiler_gym::envs::envs
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_filegroup(
  NAME custom_benchmarks_test_files
  FILES "${CMAKE_CURRENT_LIST_DIR}/invalid_ir.ll")
cg_py_test(
  NAME
    custom_benchmarks_test
  SRCS
    "custom_benchmarks_test.py"
  DATA
    compiler_gym::third_party::cbench::crc32
    ::custom_benchmarks_test_files
  DEPS
    compiler_gym::envs::envs
    compiler_gym::service::proto::proto
    compiler_gym::util::util
    tests::pytest_plugins::common
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_py_test(
  NAME
    datasets_pickle_test
  SRCS
    "datasets_pickle_test.py"
  DEPS
    compiler_gym::datasets::datasets
    compiler_gym::envs::llvm::llvm
    tests::pytest_plugins::common
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_py_test(
  NAME
    download_llvm_test
  SRCS
    "download_llvm_test.py"
  DEPS
    compiler_gym::third_party::llvm::llvm
    compiler_gym::util::util
    tests::pytest_plugins::common
    tests::test_main
)

cg_py_test(
  NAME
    episode_reward_test
  SRCS
    "episode_reward_test.py"
  DEPS
    compiler_gym::envs::envs
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_py_test(
  NAME
    fork_env_test
  SRCS
    "fork_env_test.py"
  DATA
    compiler_gym::third_party::cbench::crc32
  DEPS
    compiler_gym::envs::envs
    tests::pytest_plugins::llvm
    tests::test_main
)

py_test(
  NAME fork_regression_test
  SRCS fork_regression_test.py
  DEPS
    compiler_gym::envs::envs
    tests::test_main
    tests::pytest_plugins::llvm
)

cg_py_test(
  NAME
    fresh_environment_observation_reward_test
  SRCS
    "fresh_environment_observation_reward_test.py"
  DEPS
    compiler_gym::envs::envs
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_py_test(
  NAME
    fuzzing_regression_test
  SRCS
    "fuzzing_regression_test.py"
  DEPS
    tests::pytest_plugins::common
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_py_test(
  NAME
    gym_interface_compatability
  SRCS
    "gym_interface_compatability.py"
  DEPS
    compiler_gym::envs::llvm::llvm
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_filegroup(
  NAME invalid_ir_test_files
  FILES "${CMAKE_CURRENT_LIST_DIR}/invalid_ir.ll")
cg_py_test(
  NAME
    invalid_ir_test
  SRCS
    "invalid_ir_test.py"
  DATA
    invalid_ir_test_files
  DEPS
    compiler_gym::util::util
    tests::pytest_plugins::common
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_py_test(
  NAME
    llvm_benchmarks_test
  SRCS
    "llvm_benchmarks_test.py"
  DEPS
    compiler_gym::envs::envs
    compiler_gym::service::proto::proto
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_py_test(
  NAME
    llvm_env_test
  SRCS
    "llvm_env_test.py"
  DEPS
    compiler_gym::compiler_gym
    tests::pytest_plugins::common
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_py_test(
  NAME
    llvm_session_parameters_test
  SRCS
    "llvm_session_parameters_test.py"
  DEPS
    compiler_gym::envs::envs
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_py_test(
  NAME
    module_id_test
  SRCS
    "module_id_test.py"
  DEPS
    compiler_gym::compiler_gym
    tests::pytest_plugins::common
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_py_test(
  NAME
    multiprocessing_test
  SRCS
    "multiprocessing_test.py"
  DEPS
    compiler_gym::compiler_gym
    tests::pytest_plugins::common
    tests::test_main
)

cg_py_test(
  NAME
    observation_spaces_test
  SRCS
    "observation_spaces_test.py"
  DEPS
    compiler_gym::envs::envs
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_py_test(
  NAME
    reward_spaces_test
  SRCS
    "reward_spaces_test.py"
  DEPS
    compiler_gym::envs::envs
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_py_test(
    NAME runtime_test
    SRCS "runtime_test.py"
    DEPS
      compiler_gym::envs::llvm::llvm
      compiler_gym::service::connection
      tests::test_main
      tests::pytest_plugins::llvm
)

cg_py_test(
  NAME
    service_connection_test
  SRCS
    "service_connection_test.py"
  DEPS
    compiler_gym::compiler_gym
    compiler_gym::envs::envs
    compiler_gym::third_party::autophase::autophase
    compiler_gym::util::util
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_py_test(
  NAME
    threading_test
  SRCS
    "threading_test.py"
  DEPS
    compiler_gym::compiler_gym
    tests::test_main
)

cg_py_test(
  NAME
    validate_test
  SRCS
    "validate_test.py"
  DEPS
    compiler_gym::compiler_gym
    compiler_gym::datasets::datasets
    compiler_gym::envs::envs
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_py_test(
  NAME
    validation_regression_test
  SRCS
    "validation_regression_test.py"
  DEPS
    compiler_gym::compiler_gym
    tests::pytest_plugins::common
    tests::pytest_plugins::llvm
    tests::test_main
)
