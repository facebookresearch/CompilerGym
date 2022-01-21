# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

project(gflags)
cmake_minimum_required(VERSION 3.15)

include(ExternalProject)

ExternalProject_Add(
    cpuinfo
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/cpuinfo"
    GIT_REPOSITORY "https://github.com/pytorch/cpuinfo.git"
    GIT_TAG 2e79955ecaec85da13ac8f1245a8b2afa10d31c2
    CMAKE_ARGS
      -C "${CMAKE_CURRENT_BINARY_DIR}/cpuinfo_initial_cache.cmake"
      "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}"
)
