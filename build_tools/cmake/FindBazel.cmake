# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#[=======================================================================[.rst:
Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables in your project:

``Bazel_FOUND``
  true if Bazel is available.
``Bazel_VERSION``
  the version of Bazel.
``Bazel_EXECUTABLE``
  Path to the Bazel executable.

#]=======================================================================]

find_program(Bazel_EXECUTABLE bazel)

execute_process(COMMAND "${Bazel_EXECUTABLE}" version
  RESULT_VARIABLE _BAZEL_VERSION_EXECUTE_PROCESS_RESULT_VARIABLE
  OUTPUT_VARIABLE _BAZEL_VERSION_EXECUTE_PROCESS_OUTPUT_VARIABLE
  ERROR_QUIET
)

set(Bazel_VERSION)

if(_BAZEL_VERSION_EXECUTE_PROCESS_RESULT_VARIABLE EQUAL 0)
  string(REGEX MATCH "Build label: ([0-9a-zA-Z.]+)"
    _BAZEL_VERSION_REGEX_MATCH_OUTPUT_VARIABLE
    "${_BAZEL_VERSION_EXECUTE_PROCESS_OUTPUT_VARIABLE}"
  )

  if(CMAKE_MATCH_1)
    set(Bazel_VERSION "${CMAKE_MATCH_1}")
  endif()

  unset(_BAZEL_VERSION_REGEX_MATCH_OUTPUT_VARIABLE)
endif()

unset(_BAZEL_VERSION_EXECUTE_PROCESS_OUTPUT_VARIABLE)
unset(_BAZEL_VERSION_EXECUTE_PROCESS_RESULT_VARIABLE)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(Bazel
  FOUND_VAR Bazel_FOUND
  REQUIRED_VARS Bazel_EXECUTABLE
  VERSION_VAR Bazel_VERSION
)
