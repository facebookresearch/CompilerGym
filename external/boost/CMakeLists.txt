# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

project(protobuf)
cmake_minimum_required(VERSION 3.15)

include(ExternalProject)
include(ProcessorCount)

find_package(Git REQUIRED)

if(DEFINED CMAKE_C_FLAGS)
  list(APPEND _ADDITIONAL_ARGS "cflags=\"${CMAKE_C_FLAGS}\"")
endif()
if(DEFINED CMAKE_CXX_FLAGS)
  list(APPEND _ADDITIONAL_ARGS "cxxflags=\"${CMAKE_CXX_FLAGS}\"")
endif()
if (DEFINED CMAKE_STATIC_LINKER_FLAGS_INIT OR
  DEFINED CMAKE_SHARED_LINKER_FLAGS_INIT OR
  DEFINED CMAKE_STATIC_LINKER_FLAGS OR
  DEFINED CMAKE_SHARED_LINKER_FLAGS)
  list(APPEND _ADDITIONAL_ARGS "linkflags=\"${CMAKE_STATIC_LINKER_FLAGS_INIT} ${CMAKE_SHARED_LINKER_FLAGS_INIT} ${CMAKE_STATIC_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS}\"")
endif()

ProcessorCount(_JOBS)
ExternalProject_Add(
  boost
  PREFIX "${CMAKE_CURRENT_BINARY_DIR}/boost"
  GIT_REPOSITORY "https://github.com/boostorg/boost.git"
  GIT_TAG afb333b7c5101041f0280b2edf155c55114c9c95 #tag: boost-1.71.0
  BUILD_IN_SOURCE TRUE
  CONFIGURE_COMMAND
    "${CMAKE_COMMAND}" -E env "CC=${CMAKE_C_COMPILER}" "CXX=${CMAKE_CXX_COMPILER}"
    ./bootstrap.sh
  BUILD_COMMAND
    "${CMAKE_COMMAND}" -E env "CC=${CMAKE_C_COMPILER}" "CXX=${CMAKE_CXX_COMPILER}"
    ./b2
      -j${_JOBS}
      "--prefix=${CMAKE_INSTALL_PREFIX}"
      "--build-dir=${CMAKE_CURRENT_BINARY_DIR}/build"
      --with-filesystem
      --with-headers
      cxxstd=${CMAKE_CXX_STANDARD}
      install
  INSTALL_COMMAND ""
)
