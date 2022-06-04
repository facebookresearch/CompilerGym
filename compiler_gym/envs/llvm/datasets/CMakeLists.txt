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
    "jotaibench.py"
    "cbench.py"
    "chstone.py"
    "clgen.py"
    "csmith.py"
    "llvm_stress.py"
    "poj104.py"
  DATA
    compiler_gym::third_party::csmith::all
  DEPS
    compiler_gym::datasets::datasets
    compiler_gym::envs::llvm::llvm_benchmark
    compiler_gym::errors::errors
    compiler_gym::service::proto::proto
    compiler_gym::third_party::llvm::llvm
    compiler_gym::util::util
  PUBLIC
)
