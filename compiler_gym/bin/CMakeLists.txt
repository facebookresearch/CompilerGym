# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_library(
  NAME
    bin
  DEPS
    ::manual_env
    ::random_replay
    ::random_search
    ::service
    ::validate
  PUBLIC
)

cg_py_binary(
  NAME
    manual_env
  SRCS
    "manual_env.py"
  DEPS
    compiler_gym::envs::envs
    compiler_gym::util::util
    compiler_gym::util::flags::flags
)

cg_py_binary(
  NAME
    random_eval
  SRCS
    "random_eval.py"
  DEPS
    compiler_gym::random_search
    compiler_gym::util::util
    compiler_gym::util::flags::flags
)

cg_py_binary(
  NAME
    random_search
  SRCS
    "random_search.py"
  DEPS
    compiler_gym::random_search
    compiler_gym::util::flags::flags
)

cg_py_binary(
  NAME
    random_replay
  SRCS
    "random_replay.py"
  DEPS
    compiler_gym::random_search
    compiler_gym::util::util
    compiler_gym::util::flags::flags
)

cg_py_binary(
  NAME
    service
  SRCS
    "service.py"
  DEPS
    compiler_gym::datasets::datasets
    compiler_gym::envs::envs
    compiler_gym::spaces::spaces
    compiler_gym::util::util
    compiler_gym::util::flags::flags
)

cg_py_binary(
  NAME
    validate
  SRCS
    "validate.py"
  DEPS
    compiler_gym::util::util
    compiler_gym::util::flags::flags
    compiler_gym::validate
)
