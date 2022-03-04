# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

project(abls)
cmake_minimum_required(VERSION 3.15)

include(ExternalProject)

externalproject_add(
    absl
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/absl"
    URL
        "https://github.com/abseil/abseil-cpp/archive/ec33f404bb16564a9aea3044cd8504d6885165b0.tar.gz"
    URL_HASH
        "SHA256=20a968fab8432661b5c6f3f951cd299fe6d7677541c7dfd62407475a046c91aa"
    CMAKE_ARGS
        -C "${CMAKE_CURRENT_BINARY_DIR}/absl_initial_cache.cmake"
        "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}"
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    USES_TERMINAL_CONFIGURE TRUE
    USES_TERMINAL_BUILD TRUE
    USES_TERMINAL_INSTALL TRUE
)
