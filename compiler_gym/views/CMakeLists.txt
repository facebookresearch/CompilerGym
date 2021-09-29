# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_library(
  NAME
    views
  SRCS
    "__init__.py"
  DEPS
    ::observation
    ::reward
  PUBLIC
)

cg_py_library(
  NAME
    observation
  SRCS
    "observation.py"
  DEPS
    ::observation_space_spec
    compiler_gym::service::service
    compiler_gym::service::proto::proto
    compiler_gym::util::util
  PUBLIC
)

cg_py_library(
  NAME
    observation_space_spec
  SRCS
    "observation_space_spec.py"
  DEPS
    compiler_gym::service::service
    compiler_gym::service::proto::proto
    compiler_gym::spaces::spaces
    compiler_gym::util::util
  PUBLIC
)

cg_py_library(
  NAME
    reward
  SRCS
    "reward.py"
  DEPS
    ::observation
    compiler_gym::datasets::datasets
    compiler_gym::service::proto::proto
    compiler_gym::spaces::spaces
  PUBLIC
)
