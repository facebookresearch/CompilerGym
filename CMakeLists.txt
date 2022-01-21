# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.20)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR)
  message(FATAL_ERROR "In-source builds are unsupported. Please, build out of the source tree.")
endif()

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(DARWIN TRUE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(compiler_gym ASM C CXX)

set(CMAKE_C_STANDARD 11 CACHE STRING "C standard to be used.")
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to be used.")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_LIST_DIR}/build_tools/cmake/
)

set(COMPILER_GYM_BUILD_TESTS OFF CACHE BOOL "Enable Compiler Gym tests.")
set(COMPILER_GYM_BUILD_EXAMPLES OFF CACHE BOOL "Enable Comiler Gym examples.")

include(cg_macros)
include(cg_copts)
include(cg_genrule)
include(cg_cc_binary)
include(cg_cc_library)
include(cg_cc_test)
include(cg_py_binary)
include(cg_py_library)
include(cg_py_test)
include(cg_python)
include(cg_add_all_subdirs)
include(cg_filegroup)
include(grpc)
include(protobuf)

set(COMPILER_GYM_PYTHONPATH "$ENV{PYTHONPATH}" CACHE STRING "PYTHONPATH environment variable during build step.")
if (COMPILER_GYM_PYTHONPATH)
  string(PREPEND COMPILER_GYM_PYTHONPATH ":")
endif()
string(PREPEND COMPILER_GYM_PYTHONPATH "${CMAKE_BINARY_DIR}")
include(set_command_pythonpath)

set(DEFAULT_CMAKE_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type selected, default to ${DEFAULT_CMAKE_BUILD_TYPE}")
  set(CMAKE_BUILD_TYPE "${DEFAULT_CMAKE_BUILD_TYPE}" CACHE STRING "Build type (default ${DEFAULT_CMAKE_BUILD_TYPE})" FORCE)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

include(external/external.cmake)
add_subdirectory(compiler_gym)
if(COMPILER_GYM_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
if(COMPILER_GYM_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
