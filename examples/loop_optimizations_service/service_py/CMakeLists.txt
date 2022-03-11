# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

return()

cg_py_library(
  NAME
    loops_opt_service
  SRCS
    "loops_opt_service.py"
  DATA
    examples::loop_optimizations_service::opt_loops::opt_loops
    compiler_gym::third_party::autophase::compute_autophase
    compiler_gym::third_party::programl::compute_programl
  DEPS
  PUBLIC
)
