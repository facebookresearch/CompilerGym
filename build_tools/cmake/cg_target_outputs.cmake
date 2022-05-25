# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

include_guard(GLOBAL)
include(CMakeParseArguments)

function(cg_target_outputs)
    cmake_parse_arguments(ARG "" "RESULT" "TARGETS" ${ARGN})
    unset(RES_)
    foreach(TARGET_ ${ARG_TARGETS})
        list(APPEND RES_ $<TARGET_PROPERTY:${TARGET_},OUTPUTS>)
    endforeach()
    set("${ARG_RESULT}" ${RES_} PARENT_SCOPE)
endfunction()
