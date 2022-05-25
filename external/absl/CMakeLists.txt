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
        "https://github.com/abseil/abseil-cpp/archive/7c6608d0dbe43cf9bdf7f77787bc6bc89cc42f8b.tar.gz"
    URL_HASH
        "SHA256=971a1838a68fd175a3ba61a1f3f54561b208c46f3f27899f710fe9a454424e6b"
    CMAKE_ARGS
        -C "${CMAKE_CURRENT_BINARY_DIR}/absl_initial_cache.cmake"
        "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}"
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    USES_TERMINAL_CONFIGURE TRUE
    USES_TERMINAL_BUILD TRUE
    USES_TERMINAL_INSTALL TRUE
)
