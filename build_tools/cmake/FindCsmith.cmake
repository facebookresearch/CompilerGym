# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#[=======================================================================[.rst:
Find Csmith headers and library.

Imported Targets
^^^^^^^^^^^^^^^^

``Csmith::libcsmith``
  The Csmith library, if found.
``Csmith::csmith``
  The Csmith executable.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables in your project:

``Csmith_FOUND``
  true if Csmith is available.
``Csmith_VERSION``
  the version of Csmith.
``Csmith_ROOT_DIR``
``Csmith_EXECUTABLE``
``Csmith_LIBRARIES``
  the libraries to link against to use Csmith.
``Csmith_LIBRARY_DIRS``
  the directories of the Csmith libraries.
``Csmith_INCLUDE_DIRS``
  where to find the libinput headers.


#]=======================================================================]

include(FindPackageHandleStandardArgs)

find_program(Csmith_EXECUTABLE csmith)
if (Csmith_EXECUTABLE)
  execute_process(
    COMMAND "${Csmith_EXECUTABLE}" --version
    OUTPUT_VARIABLE Csmith_VERSION)
  string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" Csmith_VERSION "${Csmith_VERSION}")

  add_executable(Csmith::csmith IMPORTED GLOBAL)
  set_target_properties(Csmith::csmith PROPERTIES IMPORTED_LOCATION "${Csmith_EXECUTABLE}")

  get_filename_component(Csmith_ROOT_DIR "${Csmith_EXECUTABLE}" DIRECTORY)
  get_filename_component(Csmith_ROOT_DIR "${Csmith_ROOT_DIR}/.." ABSOLUTE)
  set(Csmith_ROOT_DIR "${Csmith_ROOT_DIR}" CACHE string "Path to the root installation directory of Csmith.")
endif()
find_path(Csmith_INCLUDE_DIRS csmith.h PATH_SUFFIXES csmith csmith-2.3.0)
find_library(Csmith_LIBRARIES csmith)
if (Csmith_LIBRARIES)
  get_filename_component(Csmith_LIBRARY_DIRS "${Csmith_LIBRARIES}" DIRECTORY)
endif()
if (Csmith_LIBRARIES AND Csmith_INCLUDE_DIRS)
  add_library(Csmith::libcsmith UNKNOWN IMPORTED)
  set_target_properties(Csmith::libcsmith PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${Csmith_INCLUDE_DIRS}"
    IMPORTED_LINK_INTERFACE_LANGUAGES "C"
    IMPORTED_LOCATION "${Csmith_LIBRARIES}")
endif()

find_package_handle_standard_args(Csmith
  REQUIRED_VARS
    Csmith_ROOT_DIR
    Csmith_EXECUTABLE
    Csmith_INCLUDE_DIRS
    Csmith_LIBRARIES
    Csmith_LIBRARY_DIRS
  VERSION_VAR Csmith_VERSION
  HANDLE_VERSION_RANGE)
