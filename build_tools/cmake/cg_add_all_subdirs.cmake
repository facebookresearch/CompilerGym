# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# ===
# Copied from https://github.com/google/iree/blob/main/build_tools/cmake/iree_add_all_subdirs.cmake
# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# cg_add_all_subidrs
#
# CMake function to add all subdirectories of the current directory that contain
# a CMakeLists.txt file
#
# Takes no arguments.
function(cg_add_all_subdirs)
  FILE(GLOB _CHILDREN RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
  SET(_DIRLIST "")
  foreach(_CHILD ${_CHILDREN})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${_CHILD} AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_CHILD}/CMakeLists.txt)
      LIST(APPEND _DIRLIST ${_CHILD})
    endif()
  endforeach()

  foreach(subdir ${_DIRLIST})
    add_subdirectory(${subdir})
  endforeach()
endfunction()
