# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#[=======================================================================[.rst:
Find Labm8 headers and libraries.

Imported Targets
^^^^^^^^^^^^^^^^

``Labm8::cpp::status``
``Labm8::cpp::statusor``
``Labm8::cpp::logging``
``Labm8::cpp::string``
``Labm8::cpp::stringpiece``

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables in your project:

``Labm8_FOUND``
  true if Labm8 is available.


#]=======================================================================]

include(FindPackageHandleStandardArgs)

function(has_absl _RES_VAR)
  if(TARGET absl::strings AND
    TARGET absl::time)
    set(${_RES_VAR} True PARENT_SCOPE)
  else()
    set(${_RES_VAR} False PARENT_SCOPE)
  endif()
endfunction()

function(has_fmt _RES_VAR)
  if(TARGET fmt)
    set(${_RES_VAR} True PARENT_SCOPE)
  else()
    set(${_RES_VAR} False PARENT_SCOPE)
  endif()
endfunction()

if(Labm8_FIND_REQUIRED)
  set(_REQUIRED REQUIRED)
endif()

has_absl(Labm8_HAS_absl)
if(NOT Labm8_HAS_absl)
  find_package(absl ${_REQUIRED})
  has_absl(Labm8_HAS_absl)
endif()

has_fmt(Labm8_HAS_fmt)
if(NOT Labm8_HAS_fmt)
  find_package(fmt ${_REQUIRED})
  has_fmt(Labm8_HAS_fmt)
endif()

find_path(Labm8_INCLUDE_DIRS
  labm8/cpp/status.h
  )

find_library(Labm8_cpp_string_LIBRARIES
  ${CMAKE_STATIC_LIBRARY_PREFIX}string${CMAKE_STATIC_LIBRARY_SUFFIX}
  PATH_SUFFIXES labm8/cpp)
if(Labm8_INCLUDE_DIRS AND Labm8_cpp_string_LIBRARIES)
  add_library(Labm8::cpp::string UNKNOWN IMPORTED)
  set_target_properties(Labm8::cpp::string PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${Labm8_INCLUDE_DIRS}"
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${Labm8_cpp_string_LIBRARIES}"
    IMPORTED_LINK_INTERFACE_LIBRARIES absl::strings)
endif()

find_library(Labm8_cpp_stringpiece_LIBRARIES
  ${CMAKE_STATIC_LIBRARY_PREFIX}stringpiece${CMAKE_STATIC_LIBRARY_SUFFIX}
  PATH_SUFFIXES labm8/cpp)
if(Labm8_INCLUDE_DIRS AND Labm8_cpp_stringpiece_LIBRARIES)
  add_library(Labm8::cpp::stringpiece UNKNOWN IMPORTED)
  set_target_properties(Labm8::cpp::stringpiece PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${Labm8_INCLUDE_DIRS}"
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${Labm8_cpp_stringpiece_LIBRARIES}"
    IMPORTED_LINK_INTERFACE_LIBRARIES Labm8::cpp::string)
endif()

find_library(Labm8_cpp_status_LIBRARIES
  ${CMAKE_STATIC_LIBRARY_PREFIX}status${CMAKE_STATIC_LIBRARY_SUFFIX}
  PATH_SUFFIXES labm8/cpp)
if(Labm8_INCLUDE_DIRS AND Labm8_cpp_status_LIBRARIES)
  add_library(Labm8::cpp::status UNKNOWN IMPORTED)
  set(_LINK_LIBS
    Labm8::cpp::string
    Labm8::cpp::stringpiece
    fmt)
  set_target_properties(Labm8::cpp::status PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${Labm8_INCLUDE_DIRS}"
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${Labm8_cpp_status_LIBRARIES}"
    IMPORTED_LINK_INTERFACE_LIBRARIES "${_LINK_LIBS}")
endif()

find_library(Labm8_cpp_statusor_LIBRARIES
  ${CMAKE_STATIC_LIBRARY_PREFIX}statusor${CMAKE_STATIC_LIBRARY_SUFFIX}
  PATH_SUFFIXES labm8/cpp)
if(Labm8_INCLUDE_DIRS AND Labm8_cpp_statusor_LIBRARIES)
  add_library(Labm8::cpp::statusor UNKNOWN IMPORTED)
  set_target_properties(Labm8::cpp::statusor PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${Labm8_INCLUDE_DIRS}"
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${Labm8_cpp_statusor_LIBRARIES}")
endif()

find_library(Labm8_cpp_logging_LIBRARIES
  ${CMAKE_STATIC_LIBRARY_PREFIX}logging${CMAKE_STATIC_LIBRARY_SUFFIX}
  PATH_SUFFIXES labm8/cpp)
if(Labm8_INCLUDE_DIRS AND Labm8_cpp_logging_LIBRARIES)
  add_library(Labm8::cpp::logging UNKNOWN IMPORTED)
  set(_LINK_LIBS
    Labm8::cpp::string
    Labm8::cpp::stringpiece
    absl::strings
    absl::time)
  set_target_properties(Labm8::cpp::logging PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${Labm8_INCLUDE_DIRS}"
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${Labm8_cpp_logging_LIBRARIES}"
    IMPORTED_LINK_INTERFACE_LIBRARIES "${_LINK_LIBS}")
endif()

find_package_handle_standard_args(
  Labm8
  REQUIRED_VARS
    Labm8_HAS_absl
    Labm8_HAS_fmt
    Labm8_INCLUDE_DIRS
    Labm8_cpp_string_LIBRARIES
    Labm8_cpp_stringpiece_LIBRARIES
    Labm8_cpp_status_LIBRARIES
    Labm8_cpp_statusor_LIBRARIES
    Labm8_cpp_logging_LIBRARIES)
