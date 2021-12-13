# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

project(gflags)
cmake_minimum_required(VERSION 3.15)

include(ExternalProject)

# FetchContent is not used here because for some reason the install step
# fails with not installing all necessary files in CMake v3.19.6.
ExternalProject_Add(
    gflags
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/gflags"
    GIT_REPOSITORY "https://github.com/gflags/gflags.git"
    GIT_TAG e171aa2d15ed9eb17054558e0b3a6a413bb01067 #tag v2.2.2
    CMAKE_ARGS
      -C "${CMAKE_CURRENT_BINARY_DIR}/gflags_initial_cache.cmake"
      "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}"
)
