# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#-------------------------------------------------------------------------------
# Sanitizer configurations
#-------------------------------------------------------------------------------

# Note: we add these flags to the global CMake flags, not to IREE-specific
# variables such as CMAKE_DEFAULT_COPTS so that all symbols are consistently
# defined with the same sanitizer flags, including e.g. standard library
# symbols that might be used by both IREE and non-IREE (e.g. LLVM) code.

if(${CMAKE_ENABLE_ASAN})
  string(APPEND CMAKE_CXX_FLAGS " -fsanitize=address")
  string(APPEND CMAKE_C_FLAGS " -fsanitize=address")
endif()
if(${CMAKE_ENABLE_MSAN})
  string(APPEND CMAKE_CXX_FLAGS " -fsanitize=memory")
  string(APPEND CMAKE_C_FLAGS " -fsanitize=memory")
endif()
if(${CMAKE_ENABLE_TSAN})
  string(APPEND CMAKE_CXX_FLAGS " -fsanitize=thread")
  string(APPEND CMAKE_C_FLAGS " -fsanitize=thread")
endif()
if(ANDROID)
  # Work around https://github.com/android/ndk/issues/1088
  if(${CMAKE_ENABLE_ASAN} OR ${CMAKE_ENABLE_MSAN} OR ${CMAKE_ENABLE_TSAN})
    string(APPEND CMAKE_EXE_LINKER_FLAGS " -fuse-ld=gold")
    string(APPEND CMAKE_SHARED_LINKER_FLAGS " -fuse-ld=gold")
  endif()
endif()
