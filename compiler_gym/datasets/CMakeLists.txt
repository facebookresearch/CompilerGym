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
    "benchmark.py"
    "dataset.py"
    "datasets.py"
    "files_dataset.py"
    "tar_dataset.py"
  DEPS
    ::uri
    compiler_gym::service::proto::proto
    compiler_gym::util::util
    compiler_gym::validation_result
  PUBLIC
)

cg_py_library(
  NAME
    uri
  SRCS
    "uri.py"
  PUBLIC
)
