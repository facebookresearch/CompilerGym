# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

project(gflags)
cmake_minimum_required(VERSION 3.15)

include(ExternalProject)

externalproject_add(
    cpuinfo
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/cpuinfo"
    GIT_REPOSITORY "https://github.com/pytorch/cpuinfo.git"
    GIT_TAG de2fa78ebb431db98489e78603e4f77c1f6c5c57
    CMAKE_ARGS
        -C "${CMAKE_CURRENT_BINARY_DIR}/cpuinfo_initial_cache.cmake"
        "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}"
    USES_TERMINAL_CONFIGURE TRUE
    USES_TERMINAL_BUILD TRUE
    USES_TERMINAL_INSTALL TRUE
)
