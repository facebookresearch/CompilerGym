# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

project(gflags)
cmake_minimum_required(VERSION 3.15)

include(ExternalProject)

include(ProcessorCount)
if (DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL})
  set(_JOBS $ENV{CMAKE_BUILD_PARALLEL_LEVEL})
else()
  ProcessorCount(_JOBS)
endif()

# LLVM is memory hungry during linking
cmake_host_system_information(RESULT _RAM QUERY AVAILABLE_PHYSICAL_MEMORY)
set(_RAM_PER_LINK_JOB 10240) # 10 GiB
math(EXPR _LINK_JOBS "${_RAM} / ${_RAM_PER_LINK_JOB}" OUTPUT_FORMAT DECIMAL)
if (NOT _LINK_JOBS)
  set(_LINK_JOBS 1)
endif()

# The source is still required even if version is external.
if(NOT COMPILER_GYM_LLVM_PROVIDER STREQUAL "internal")
  set(_BUILD_COMMAND "")
  set(_INSTALL_COMMAND "")
else()
  set(_BUILD_COMMAND "${CMAKE_COMMAND}" --build "<BINARY_DIR>")
  set(_INSTALL_COMMAND "${CMAKE_COMMAND}" --install "<BINARY_DIR>")
endif()

file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/llvm_initial_cache.cmake" "set(LLVM_ENABLE_PROJECTS \"clang;polly\" CACHE STRING \"\")\n")

ExternalProject_Add(
  llvm
  PREFIX "${CMAKE_CURRENT_BINARY_DIR}/llvm"
  GIT_REPOSITORY "https://github.com/llvm/llvm-project.git"
  GIT_TAG d32170dbd5b0d54436537b6b75beaf44324e0c28 #tag llvmorg-10.0.0
  SOURCE_SUBDIR llvm
  CMAKE_ARGS
    -G Ninja
    -C "${CMAKE_CURRENT_BINARY_DIR}/llvm_initial_cache.cmake"
    "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}"
    -DLLVM_ABI_BREAKING_CHECKS=FORCE_OFF
    -DLLVM_PARALLEL_LINK_JOBS=${_LINK_JOBS}
    -DLLVM_PARALLEL_COMPILE_JOBS=${_JOBS}
    -DLLVM_TARGETS_TO_BUILD=host
  BUILD_COMMAND ${_BUILD_COMMAND}
  INSTALL_COMMAND ${_INSTALL_COMMAND}
)
