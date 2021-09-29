# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

project(gflags)
cmake_minimum_required(VERSION 3.15)

include(ExternalProject)

ExternalProject_Add(
  csmith
  PREFIX "${CMAKE_CURRENT_BINARY_DIR}/csmith"
  GIT_REPOSITORY "https://github.com/csmith-project/csmith.git"
  GIT_TAG 30dccd73b78652c4719f36572994778a5b233a4e #tag csmith-2.3.0
  CMAKE_ARGS
    -C "${CMAKE_CURRENT_BINARY_DIR}/csmith_initial_cache.cmake"
    "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}"
    "-DCOMPILE_DEFINITIONS=_LIBCPP_ENABLE_CXX17_REMOVED_FEATURES"
)
