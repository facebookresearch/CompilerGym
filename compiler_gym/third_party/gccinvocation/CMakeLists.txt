# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Library for parsing compiler command line invocations.
#
# https://github.com/fedora-static-analysis/gccinvocation

cg_add_all_subdirs()

cg_py_library(
  NAME
    gccinvocation
  SRCS
    "gccinvocation.py"
  PUBLIC
)
