# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

project(abls)
cmake_minimum_required(VERSION 3.15)

include(ExternalProject)

ExternalProject_Add(
    absl
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/absl"
    URL "https://github.com/abseil/abseil-cpp/archive/997aaf3a28308eba1b9156aa35ab7bca9688e9f6.tar.gz"
    URL_HASH "SHA256=35f22ef5cb286f09954b7cc4c85b5a3f6221c9d4df6b8c4a1e9d399555b366ee"
    CMAKE_ARGS
      -C "${CMAKE_CURRENT_BINARY_DIR}/absl_initial_cache.cmake"
      "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}"
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
)
