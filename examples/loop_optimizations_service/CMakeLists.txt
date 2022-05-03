# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

if(NOT COMPILER_GYM_ENABLE_LLVM_ENV)
    return()
endif()

cg_add_all_subdirs()
