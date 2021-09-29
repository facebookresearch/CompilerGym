# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#[=======================================================================[.rst:
Find Subprocess headers and library.

Imported Targets
^^^^^^^^^^^^^^^^

``Subprocess::libsubprocess``

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables in your project:

``Subprocess_FOUND``
  true if Subprocess is available.


#]=======================================================================]

include(FindPackageHandleStandardArgs)

find_path(Subprocess_INCLUDE_DIRS subprocess/subprocess.hpp)
if (Subprocess_INCLUDE_DIRS)
  add_library(Subprocess::libsubprocess INTERFACE IMPORTED)
  set_target_properties(Subprocess::libsubprocess PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${Subprocess_INCLUDE_DIRS}")
endif()


find_package_handle_standard_args(Subprocess
  REQUIRED_VARS
    Subprocess_INCLUDE_DIRS)
