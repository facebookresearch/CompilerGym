# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_library(
  NAME
    llvm
  SRCS
    "__init__.py"
  DATA
    compiler_gym::envs::llvm::service::service
  DEPS
    ::compute_observation
    ::llvm_benchmark
    ::llvm_env
    compiler_gym::util::util
  PUBLIC
)

cg_py_library(
  NAME compute_observation
  SRCS compute_observation.py
  DATA compiler_gym::envs::llvm::service::compute_observation-files
  DEPS compiler_gym::util::util
)

cg_py_library(
  NAME
    llvm_benchmark
  SRCS
    "llvm_benchmark.py"
  DEPS
    compiler_gym::datasets::datasets
    compiler_gym::service::proto::proto
    compiler_gym::third_party::llvm::llvm
    compiler_gym::util::util
  PUBLIC
)

cg_py_library(
  NAME
    llvm_env
  SRCS
    "llvm_env.py"
  DEPS
    ::llvm_benchmark
    ::llvm_rewards
    compiler_gym::datasets::datasets
    compiler_gym::envs::compiler_env
    compiler_gym::envs::llvm::datasets::datasets
    compiler_gym::spaces::spaces
    compiler_gym::third_party::autophase::autophase
    compiler_gym::third_party::inst2vec::inst2vec
    compiler_gym::third_party::llvm::llvm
    compiler_gym::third_party::llvm::instcount
  PUBLIC
)

cg_py_library(
  NAME
    llvm_rewards
  SRCS
    "llvm_rewards.py"
  DEPS
    compiler_gym::service::service
    compiler_gym::spaces::spaces
    compiler_gym::util::util
    compiler_gym::views::views
  PUBLIC
)

string(CONCAT _CMD
  "\"${Python3_EXECUTABLE}\" "
    "\"$<TARGET_PROPERTY:compiler_gym__envs__llvm__make_specs,LOCATION>\" "
    "\"$<TARGET_FILE:compiler_gym__envs__llvm__service__compiler_gym-llvm-service>\" "
    "\"$@\"")
set_command_pythonpath(COMMAND "${_CMD}" RESULT _CMD)
cg_genrule(
  NAME specs
  OUTS "specs.py"
  COMMAND "${_CMD}"
  DEPENDS
    ::make_specs
    compiler_gym::compiler_gym_partial
    compiler_gym::envs::llvm::service::service
    compiler_gym::envs::llvm::service::compiler_gym-llvm-service
)

cg_py_binary(
  NAME
    make_specs
  SRCS
    "make_specs.py"
  DATA "${CMAKE_CURRENT_BINARY_DIR}/service/passes/flag_descriptions.txt"
  DEPS
    ::llvm_env
    compiler_gym::util::util
)
