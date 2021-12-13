# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

include(CMakeParseArguments)

function(set_command_pythonpath)
  cmake_parse_arguments(
    _ARG
    ""
    "COMMAND;RESULT"
    ""
    ${ARGN}
  )

  if(COMPILER_GYM_PYTHONPATH)
    set(${_ARG_RESULT} "\"${CMAKE_COMMAND}\" -E env \"PYTHONPATH=${COMPILER_GYM_PYTHONPATH}\" ${_ARG_COMMAND}" PARENT_SCOPE)
  else()
    set(${_ARG_RESULT} ${_ARG_COMMAND} PARENT_SCOPE)
  endif()

endfunction()
