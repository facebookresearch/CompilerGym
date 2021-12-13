# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

# This target trickery with compiler_gym and compiler_gym_partial
# is needed because specs.py imports the compiler_gym module,
# therefore creating a circular dependency.
# compiler_gym_partial is all the other bits of the proto package so that
# specs.py can import it.
cg_py_library(
  NAME
    compiler_gym
  GENERATED_SRCS
    "$<TARGET_PROPERTY:compiler_gym__envs__llvm__specs,LOCATION>"
  DEPS
    ::compiler_gym_partial
    compiler_gym::envs::llvm::specs
  PUBLIC
)

cg_py_library(
  NAME
    compiler_gym_partial
  SRCS
    "__init__.py"
  DEPS
    ::random_replay
    ::random_search
    ::validate
    compiler_gym::bin::bin
    compiler_gym::datasets::datasets
    compiler_gym::envs::envs
    compiler_gym::leaderboard::leaderboard
    compiler_gym::service::service
    compiler_gym::spaces::spaces
    compiler_gym::util::util
    compiler_gym::util::flags::flags
    compiler_gym::wrappers::wrappers
  PUBLIC
)

cg_py_library(
  NAME
    compiler_env_state
  SRCS
    "compiler_env_state.py"
  DEPS
    compiler_gym::datasets::uri
    compiler_gym::util::util
  PUBLIC
)

cg_py_library(
  NAME
    random_replay
  SRCS
    "random_replay.py"
  DEPS
    ::random_search
    compiler_gym::envs::envs
    compiler_gym::util::util
  PUBLIC
)

cg_py_library(
  NAME
    random_search
  SRCS
    "random_search.py"
  DATA
    compiler_gym::envs::llvm::service::service
  DEPS
    compiler_gym::envs::envs
    compiler_gym::service::connection
    compiler_gym::util::util
  PUBLIC
)

cg_py_library(
  NAME
    validate
  SRCS
    "validate.py"
  DEPS
    ::validation_error
    ::validation_result
    compiler_gym::envs::compiler_env
    compiler_gym::spaces::spaces
    compiler_gym::util::util
  PUBLIC
)

cg_py_library(
  NAME
    validation_error
  SRCS
    "validation_error.py"
  PUBLIC
)

cg_py_library(
  NAME
    validation_result
  SRCS
    "validation_result.py"
  DEPS
    ::compiler_env_state
    ::validation_error
    compiler_gym::util::util
  PUBLIC
)
