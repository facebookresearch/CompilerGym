# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_library(
  NAME
    wrappers
  SRCS
    "__init__.py"
    "commandline.py"
    "core.py"
    "datasets.py"
    "llvm.py"
    "time_limit.py"
    "validation.py"
  DEPS
    compiler_gym::datasets::datasets
    compiler_gym::envs::envs
    compiler_gym::util::util
    compiler_gym::views::views
  PUBLIC
)
