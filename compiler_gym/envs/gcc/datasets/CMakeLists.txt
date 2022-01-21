# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_library(
  NAME
    datasets
  SRCS
    "__init__.py"
    "anghabench.py"
    "chstone.py"
    "csmith.py"
  DATA
    compiler_gym::third_party::csmith::all
  DEPS
    compiler_gym::datasets::datasets
    compiler_gym::service::proto::proto
    compiler_gym::util::util
  PUBLIC
)
