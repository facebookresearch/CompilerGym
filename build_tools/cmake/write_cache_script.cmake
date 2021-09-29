# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

include_guard(GLOBAL)

function(write_cache_script _DST_FILE)
  file(WRITE "${_DST_FILE}" "")
  set(_VARS
    CMAKE_BUILD_TYPE
    CMAKE_GENERATOR
    CMAKE_C_COMPILER
    CMAKE_CXX_COMPILER
    CMAKE_CXX_STANDARD
    CMAKE_CXX_FLAGS
    CMAKE_C_FLAGS
    CMAKE_GENERATOR_TOOLSET
    CMAKE_GENERATOR_PLATFORM
    CMAKE_C_COMPILER_LAUNCHER
    CMAKE_CXX_COMPILER_LAUNCHER
    CMAKE_MODULE_LINKER_FLAGS_INIT
    CMAKE_MODULE_LINKER_FLAGS
    CMAKE_STATIC_LINKER_FLAGS_INIT
    CMAKE_STATIC_LINKER_FLAGS
    CMAKE_SHARED_LINKER_FLAGS_INIT
    CMAKE_SHARED_LINKER_FLAGS
    CMAKE_EXE_LINKER_FLAGS_INIT
    CMAKE_EXE_LINKER_FLAGS
  )
  foreach(_VAR in ${_VARS})
    if(DEFINED ${_VAR})
      file(APPEND "${_DST_FILE}" "set(${_VAR} \"${${_VAR}}\" CACHE STRING \"\")\n")
    endif()
  endforeach()
endfunction()
