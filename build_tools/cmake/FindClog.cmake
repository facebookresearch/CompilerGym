# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#[=======================================================================[.rst:
Find Clog headers and libraries.

Imported Targets
^^^^^^^^^^^^^^^^

``Clog::libclog``

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables in your project:

``Clog_FOUND``
  true if Clog is available.


#]=======================================================================]

include(FindPackageHandleStandardArgs)

find_path(Clog_INCLUDE_DIRS clog.h
  PATH_SUFFIXES include)

find_library(Clog_LIBRARIES clog PATH_SUFFIXES lib)
if(Clog_INCLUDE_DIRS AND Clog_LIBRARIES)
  add_library(Clog::libclog UNKNOWN IMPORTED)
  set_target_properties(Clog::libclog PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${Clog_INCLUDE_DIRS}"
    IMPORTED_LINK_INTERFACE_LANGUAGES "C"
    IMPORTED_LOCATION "${Clog_LIBRARIES}")
endif()
find_package_handle_standard_args(
  Clog
  REQUIRED_VARS
    Clog_INCLUDE_DIRS
    Clog_LIBRARIES)
