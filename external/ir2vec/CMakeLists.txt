# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

project(ir2vec)
cmake_minimum_required(VERSION 3.15)

include(ExternalProject)

externalproject_add(
    ir2vec
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/ir2vec"
    GIT_REPOSITORY "https://github.com/ChrisCummins/IR2Vec.git"
    GIT_TAG 828e50584b9c8bc305208e22d2cca272bdb1ab64
    SOURCE_SUBDIR "src"
    CMAKE_ARGS
        -C "${CMAKE_CURRENT_BINARY_DIR}/ir2vec_initial_cache.cmake"
        "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}"
        "-DLT_LLVM_INSTALL_DIR=${CMAKE_INSTALL_PREFIX}"
    USES_TERMINAL_CONFIGURE TRUE
    USES_TERMINAL_BUILD TRUE
    USES_TERMINAL_INSTALL TRUE
)
