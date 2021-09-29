# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#[=======================================================================[.rst:
Find ProGraML headers and libraries.

Imported Targets
^^^^^^^^^^^^^^^^

``ProGraML::graph::format::node_link_graph``
``ProGraML::ir::llvm::llvm-10``
``ProGraML::proto::programl_cc``
``ProGraML::graph::program_graph_builder``

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables in your project:

``ProGraML_FOUND``
  true if ProGraML is available.


#]=======================================================================]

include(FindPackageHandleStandardArgs)

function(has_Labm8 _RES_VAR)
  if(TARGET Labm8::cpp::status AND
    TARGET Labm8::cpp::statusor AND
    TARGET Labm8::cpp::logging AND
    TARGET Labm8::cpp::string AND
    TARGET Labm8::cpp::stringpiece)
    set(${_RES_VAR} True PARENT_SCOPE)
  else()
    set(${_RES_VAR} False PARENT_SCOPE)
  endif()
endfunction()

function(has_absl _RES_VAR)
  if(TARGET absl::flat_hash_map AND
  TARGET absl::flat_hash_set)
    set(${_RES_VAR} True PARENT_SCOPE)
  else()
    set(${_RES_VAR} False PARENT_SCOPE)
  endif()
endfunction()

if(ProGraML_FIND_REQUIRED)
  set(_REQUIRED REQUIRED)
endif()

has_Labm8(ProGraML_HAS_Labm8)
if(NOT ProGraML_HAS_Labm8)
  find_package(Labm8 ${_REQUIRED})
  has_Labm8(ProGraML_HAS_Labm8)
endif()

has_absl(ProGraML_HAS_absl)
if(NOT ProGraML_HAS_absl)
  find_package(absl ${_REQUIRED})
  has_absl(ProGraML_HAS_absl)
endif()

# Deliberately find static libs.
# For some reason the linker takes the path to the library
# instead of just the name for the dynamic section when linking to these libs.
# See https://stackoverflow.com/questions/70088552/linker-adds-the-path-to-library-in-the-dynamic-section-instead-of-its-name
find_library(ProGraML_proto_programl_cc_LIBRARIES
  ${CMAKE_STATIC_LIBRARY_PREFIX}programl${CMAKE_STATIC_LIBRARY_SUFFIX}
  PATH_SUFFIXES programl/proto)
find_path(ProGraML_proto_programl_cc_INCLUDE_DIRS programl/proto/program_graph_options.pb.h)
if (ProGraML_proto_programl_cc_LIBRARIES AND ProGraML_proto_programl_cc_INCLUDE_DIRS)
  add_library(ProGraML::proto::programl_cc UNKNOWN IMPORTED)
  set_target_properties(ProGraML::proto::programl_cc PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ProGraML_proto_programl_cc_INCLUDE_DIRS}"
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${ProGraML_proto_programl_cc_LIBRARIES}")
endif()

find_library(ProGraML_graph_program_graph_builder_LIBRARIES
  ${CMAKE_STATIC_LIBRARY_PREFIX}program_graph_builder${CMAKE_STATIC_LIBRARY_SUFFIX}
  PATH_SUFFIXES programl/graph/)
find_path(ProGraML_graph_program_graph_builder_INCLUDE_DIRS programl/graph/program_graph_builder.h)
if (ProGraML_graph_program_graph_builder_LIBRARIES AND
  ProGraML_graph_program_graph_builder_INCLUDE_DIRS)
  set(_INCLUDE_DIRS ${ProGraML_graph_program_graph_builder_INCLUDE_DIRS})
  add_library(ProGraML::graph::program_graph_builder UNKNOWN IMPORTED)
  set_target_properties(ProGraML::graph::program_graph_builder PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES
    "${ProGraML_graph_program_graph_builder_INCLUDE_DIRS}")
  set_target_properties(ProGraML::graph::program_graph_builder PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${ProGraML_graph_program_graph_builder_LIBRARIES}")
  set(_LINK_LIBS
    ProGraML::proto::programl_cc
    absl::flat_hash_map
    absl::flat_hash_set
    Labm8::cpp::logging
    Labm8::cpp::status
    Labm8::cpp::statusor
    Labm8::cpp::string)
  set_target_properties(ProGraML::graph::program_graph_builder
    PROPERTIES IMPORTED_LINK_INTERFACE_LIBRARIES "${_LINK_LIBS}")
endif()

find_library(ProGraML_graph_features_LIBRARIES
  ${CMAKE_STATIC_LIBRARY_PREFIX}features${CMAKE_STATIC_LIBRARY_SUFFIX}
  PATH_SUFFIXES programl/graph/)
find_path(ProGraML_graph_features_INCLUDE_DIRS programl/graph/features.h)
if (ProGraML_graph_features_LIBRARIES AND
  ProGraML_graph_features_INCLUDE_DIRS)
  set(_INCLUDE_DIRS ${ProGraML_graph_features_INCLUDE_DIRS})
  add_library(ProGraML::graph::features UNKNOWN IMPORTED)
  set_target_properties(ProGraML::graph::features PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ProGraML_graph_features_INCLUDE_DIRS}"
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${ProGraML_graph_features_LIBRARIES}"
    IMPORTED_LINK_INTERFACE_LIBRARIES ProGraML::proto::programl_cc)
endif()

find_library(ProGraML_graph_format_node_link_graph_LIBRARIES
  ${CMAKE_STATIC_LIBRARY_PREFIX}node_link_graph${CMAKE_STATIC_LIBRARY_SUFFIX}
  PATH_SUFFIXES programl/graph/format)
find_path(ProGraML_graph_format_node_link_graph_INCLUDE_DIRS programl/graph/format/node_link_graph.h)
if (ProGraML_graph_format_node_link_graph_LIBRARIES AND
  ProGraML_graph_format_node_link_graph_INCLUDE_DIRS)
  add_library(ProGraML::graph::format::node_link_graph UNKNOWN IMPORTED)
  set(_INCLUDE_DIRS ${ProGraML_graph_format_node_link_graph_INCLUDE_DIRS})
  set(_LINK_LIBS
    ProGraML::proto::programl_cc
    Labm8::cpp::status
    Labm8::cpp::logging)
  set_target_properties(ProGraML::graph::format::node_link_graph PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_INCLUDE_DIRS}"
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${ProGraML_graph_format_node_link_graph_LIBRARIES}"
    IMPORTED_LINK_INTERFACE_LIBRARIES "${_LINK_LIBS}"
    )
endif()

find_library(ProGraML_ir_llvm_llvm_10_LIBRARIES
  ${CMAKE_STATIC_LIBRARY_PREFIX}llvm-10${CMAKE_STATIC_LIBRARY_SUFFIX}
  PATH_SUFFIXES programl/ir/llvm)
find_path(ProGraML_ir_llvm_llvm_10_INCLUDE_DIRS programl/ir/llvm/llvm.h)
if (ProGraML_ir_llvm_llvm_10_LIBRARIES AND ProGraML_ir_llvm_llvm_10_INCLUDE_DIRS)
  add_library(ProGraML::ir::llvm::llvm-10 UNKNOWN IMPORTED)
  set(_LINK_LIBS
    ProGraML::graph::features
    ProGraML::graph::program_graph_builder
    ProGraML::proto::programl_cc
    absl::flat_hash_map
    absl::flat_hash_set
    Labm8::cpp::status
    Labm8::cpp::statusor
    Labm8::cpp::string)
  set_target_properties(ProGraML::ir::llvm::llvm-10 PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ProGraML_ir_llvm_llvm_10_INCLUDE_DIRS}"
    IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${ProGraML_ir_llvm_llvm_10_LIBRARIES}"
    IMPORTED_LINK_INTERFACE_LIBRARIES "${_LINK_LIBS}")
endif()

find_package_handle_standard_args(ProGraML
  REQUIRED_VARS
    ProGraML_HAS_Labm8
    ProGraML_HAS_absl
    ProGraML_graph_format_node_link_graph_LIBRARIES
    ProGraML_graph_format_node_link_graph_INCLUDE_DIRS
    ProGraML_graph_features_LIBRARIES
    ProGraML_graph_features_INCLUDE_DIRS
    ProGraML_graph_program_graph_builder_INCLUDE_DIRS
    ProGraML_graph_program_graph_builder_LIBRARIES
    ProGraML_ir_llvm_llvm_10_LIBRARIES
    ProGraML_ir_llvm_llvm_10_INCLUDE_DIRS
    ProGraML_proto_programl_cc_LIBRARIES
    ProGraML_proto_programl_cc_INCLUDE_DIRS)
