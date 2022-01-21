# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

function(cg_filegroup)
  cmake_parse_arguments(
    _ARG
    "PUBLIC"
    "NAME"
    "FILES;DEPENDS"
    ${ARGN}
  )
  rename_bazel_targets(_NAME "${_ARG_NAME}")
  add_custom_target(${_NAME})

  foreach(FILE_ ${_ARG_FILES})
    if(IS_ABSOLUTE "${FILE_}")
      set(_INPUT_PATH "${FILE_}")
      get_filename_component(_FILE_NAME ${FILE_} NAME)
      canonize_bazel_target_names(_FILE_TARGET "${_FILE_NAME}")
      rename_bazel_targets(_TARGET "${_FILE_TARGET}")
      set(_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${_FILE_NAME}")
    else()
      canonize_bazel_target_names(_FILE_TARGET "${FILE_}")
      rename_bazel_targets(_TARGET "${_FILE_TARGET}")
      set(_INPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${FILE_}")
      set(_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${FILE_}")
    endif()

    if(NOT TARGET ${_TARGET})
      if (NOT _INPUT_PATH STREQUAL _OUTPUT_PATH)
        add_custom_command(OUTPUT "${_OUTPUT_PATH}"
          COMMAND ${CMAKE_COMMAND} -E create_symlink "${_INPUT_PATH}" "${_OUTPUT_PATH}"
          DEPENDS "${_INPUT_PATH}")
      endif()
      add_custom_target(${_TARGET} DEPENDS "${_OUTPUT_PATH}")
    endif()

    add_dependencies(${_NAME} ${_TARGET})
  endforeach()

  if(_ARG_DEPENDS)
    rename_bazel_targets(_DEPS "${_ARG_DEPENDS}")
    add_dependencies(${_NAME} ${_DEPS})
  endif()

  set_target_properties(${_NAME} PROPERTIES
    IS_FILEGROUP TRUE
    OUTPUTS "${_SRCS}")
endfunction()
