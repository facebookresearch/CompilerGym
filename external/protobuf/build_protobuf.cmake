# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.15)

# Trickery to circumvent https://gitlab.kitware.com/cmake/cmake/-/issues/19703
# Avoids rebuilding if the git was in the same state as in the previous build.

execute_process(
  COMMAND "${GIT_EXECUTABLE}" log -n 1 --pretty=format:%H
  WORKING_DIRECTORY "${GIT_REPOSITORY_DIR}"
  OUTPUT_VARIABLE _GIT_HASH
  COMMAND_ERROR_IS_FATAL ANY
  )

execute_process(
  COMMAND "${GIT_EXECUTABLE}" diff --quiet
  WORKING_DIRECTORY "${GIT_REPOSITORY_DIR}"
  RESULT_VARIABLE _GIT_DIFF_RES)
if(_GIT_DIFF_RES STREQUAL 0)
  set(_IS_GIT_DIRTY FALSE)
else()
  set(_IS_GIT_DIRTY TRUE)
endif()

if(NOT _IS_GIT_DIRTY AND EXISTS "${GIT_REPOSITORY_DIR}/../build_git_hash" AND
  "${GIT_REPOSITORY_DIR}/../build_git_hash" IS_NEWER_THAN "${CMAKE_CURRENT_LIST_DIR}")
  file(READ "${GIT_REPOSITORY_DIR}/../build_git_hash" _PREV_GIT_HASH)
  if (_GIT_HASH STREQUAL _PREV_GIT_HASH)
    return()
  endif()
endif()

file(REMOVE "${GIT_REPOSITORY_DIR}/../build_git_hash")

execute_process(
  COMMAND ./autogen.sh
  WORKING_DIRECTORY "${GIT_REPOSITORY_DIR}"
  COMMAND_ERROR_IS_FATAL ANY)

execute_process(
  COMMAND "${CMAKE_COMMAND}"
    -E env
      "CC=${CMAKE_C_COMPILER}"
      "CXX=${CMAKE_CXX_COMPILER}"
      "CFLAGS=${CMAKE_C_FLAGS} $ENV{CFLAGS}"
      "CXXFLAGS=-std=c++${CMAKE_CXX_STANDARD} ${CMAKE_CXX_FLAGS} $ENV{CFLAGS}"
      "LDFLAGS=-Wl,-rpath,${CMAKE_INSTALL_PREFIX}/lib ${CMAKE_STATIC_LINKER_FLAGS_INIT} ${CMAKE_SHARED_LINKER_FLAGS_INIT} ${CMAKE_EXE_LINKER_FLAGS_INIT} ${CMAKE_STATIC_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS} $ENV{LDFLAGS}"
      ./configure
        "--prefix=${CMAKE_INSTALL_PREFIX}"
        # --enable-shared=no seems to not work. It still produces shared libraries.
        "--enable-shared=no"
  WORKING_DIRECTORY "${GIT_REPOSITORY_DIR}"
  COMMAND_ERROR_IS_FATAL ANY)


execute_process(
  COMMAND "${CMAKE_COMMAND}" -E remove_directory "${CMAKE_INSTALL_PREFIX}"
  WORKING_DIRECTORY "${GIT_REPOSITORY_DIR}"
  COMMAND_ERROR_IS_FATAL ANY)

include(ProcessorCount)
ProcessorCount(_JOBS)
execute_process(
  COMMAND make -j${_JOBS} install
  WORKING_DIRECTORY "${GIT_REPOSITORY_DIR}"
  COMMAND_ERROR_IS_FATAL ANY)

if(_IS_GIT_DIRTY)
  return()
endif()

file(WRITE "${GIT_REPOSITORY_DIR}/../build_git_hash" "${_GIT_HASH}")
