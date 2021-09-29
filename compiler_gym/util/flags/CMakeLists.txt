# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_library(
  NAME
    flags
  SRCS
    "benchmark_from_flags.py"
    "env_from_flags.py"
    "episode_length.py"
    "episodes.py"
    "learning_rate.py"
    "nproc.py"
    "output_dir.py"
    "seed.py"
  DEPS
    compiler_gym::datasets::datasets
    compiler_gym::envs::envs
    compiler_gym::service::service
    compiler_gym::service::proto::proto
  PUBLIC
)
