include_guard(GLOBAL)
include(CMakeParseArguments)
include(write_cache_script)

function(build_external_cmake_project)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;SRC_DIR"
    "CONFIG_ARGS"
    ${ARGN}
  )

  set(_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}/external/${_RULE_NAME}")
  set(_INTIAL_CACHE_PATH "${CMAKE_CURRENT_BINARY_DIR}/external/${_RULE_NAME}/${_RULE_NAME}_initial_cache.cmake")
  write_cache_script("${_INTIAL_CACHE_PATH}")
  execute_process(
    COMMAND "${CMAKE_COMMAND}"
    -C "${_INTIAL_CACHE_PATH}"
    -S "${_RULE_SRC_DIR}"
    -B "${_BIN_DIR}"
    -D "CMAKE_INSTALL_PREFIX=${_BIN_DIR}/install"
    ${_RULES_CONFIG_ARGS}
    COMMAND_ERROR_IS_FATAL ANY
  )
  execute_process(
    COMMAND
    "${CMAKE_COMMAND}"
    --build "${_BIN_DIR}"
    COMMAND_ERROR_IS_FATAL ANY
  )
  execute_process(
    COMMAND
    "${CMAKE_COMMAND}"
    --install "${_BIN_DIR}"
    COMMAND_ERROR_IS_FATAL ANY
  )
  list(PREPEND CMAKE_PREFIX_PATH "${_BIN_DIR}/install")
  set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} PARENT_SCOPE)
endfunction()
