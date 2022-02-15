# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

set(_DEPS "compiler_gym-llvm-service")
if(DARWIN)
  #list(APPEND _DEPS "@llvm//:darwin")
endif()
cg_filegroup(
  NAME "service"
  DEPENDS ${_DEPS}
)

cg_cc_binary(
  NAME
    compiler_gym-llvm-service
  SRCS
    "RunService.cc"
  DEPS
    ::LlvmSession
    compiler_gym::service::runtime::cc_runtime
)

cg_cc_library(
  NAME
    ActionSpace
  HDRS
    "ActionSpace.h"
    "$<TARGET_PROPERTY:compiler_gym__envs__llvm__service__passes__actions_genfiles,BINARY_DIR>/ActionEnum.h"
  SRCS
    "ActionSpace.cc"
  DEPS
    compiler_gym::service::proto::compiler_gym_service_cc
    compiler_gym::util::EnumUtil
    compiler_gym::util::Unreachable
  ABS_DEPS
    fmt
    magic_enum
  NON_LIB_DEPS
    compiler_gym::envs::llvm::service::passes::actions_genfiles
  PUBLIC
)

llvm_map_components_to_libnames(_LLVM_LIBS core support bitwriter)
cg_cc_library(
  NAME
    Benchmark
  HDRS
    "Benchmark.h"
  SRCS
    "Benchmark.cc"
  DEPS
    ::BenchmarkDynamicConfig
    ::Cost
    compiler_gym::service::proto::compiler_gym_service_cc
    compiler_gym::util::GrpcStatusMacros
    compiler_gym::util::RunfilesPath
    compiler_gym::util::Subprocess
  ABS_DEPS
    Boost::filesystem
    grpc++
    fmt
    glog::glog
    ${_LLVM_LIBS}
  INCLUDES
    ${LLVM_INCLUDE_DIRS}
  DEFINES
    ${LLVM_DEFINITIONS}
  PUBLIC
)

cg_cc_library(
  NAME
    BenchmarkDynamicConfig
  HDRS
    "BenchmarkDynamicConfig.h"
  SRCS
    "BenchmarkDynamicConfig.cc"
  DEPS
    compiler_gym::service::proto::compiler_gym_service_cc
    compiler_gym::util::RunfilesPath
    compiler_gym::util::Subprocess
  ABS_DEPS
    Boost::filesystem
)

llvm_map_components_to_libnames(_LLVM_LIBS core)
cg_cc_library(
  NAME
    BenchmarkFactory
  HDRS
    "BenchmarkFactory.h"
  SRCS
    "BenchmarkFactory.cc"
  DEPS
    ::Benchmark
    ::Cost
    compiler_gym::service::proto::compiler_gym_service_cc
    compiler_gym::util::GrpcStatusMacros
    compiler_gym::util::RunfilesPath
    compiler_gym::util::StrLenConstexpr
  ABS_DEPS
    Boost::filesystem
    grpc++
    fmt
    glog::glog
    ${_LLVM_LIBS}
  INCLUDES
    ${LLVM_INCLUDE_DIRS}
  DEFINES
    ${LLVM_DEFINITIONS}
  PUBLIC
)

llvm_map_components_to_libnames(_LLVM_LIBS core support irreader)
cg_cc_binary(
  NAME compute_observation
  SRCS ComputeObservation.cc
  COPTS
    "-DGOOGLE_PROTOBUF_NO_RTTI"
    "-fno-rtti"
  DEPS
    ::BenchmarkFactory
    ::Observation
    ::ObservationSpaces
    compiler_gym::service::proto::compiler_gym_service_cc
  ABS_DEPS
    Boost::filesystem
    glog::glog
    ${_LLVM_LIBS}
    magic_enum
  INCLUDES
    ${LLVM_INCLUDE_DIRS}
  DEFINES
    ${LLVM_DEFINITIONS}
)

set(_FILES "${CMAKE_CURRENT_BINARY_DIR}/compute_observation")
if(DARWIN)
  message(FATAL_ERROR "TODO(boian): implement")
endif()
cg_filegroup(
  NAME compute_observation-files
  FILES ${_FILES}
)

llvm_map_components_to_libnames(_LLVM_LIBS core transformutils ipo)
cg_cc_library(
  NAME
    Cost
  HDRS
    "Cost.h"
  SRCS
    "Cost.cc"
  DEPS
    compiler_gym::util::GrpcStatusMacros
    compiler_gym::util::RunfilesPath
    compiler_gym::util::Subprocess
    compiler_gym::util::Unreachable
  ABS_DEPS
    Boost::filesystem
    Boost::headers
    grpc++
    fmt
    glog::glog
    ${_LLVM_LIBS}
    magic_enum
  INCLUDES
    ${LLVM_INCLUDE_DIRS}
  DEFINES
    ${LLVM_DEFINITIONS}
  PUBLIC
)

llvm_map_components_to_libnames(_LLVM_LIBS
  core analysis coroutines objcarcopts target codegen
  x86codegen x86asmparser #TODO(boian): can these be found programmatically
  )
cg_cc_library(
  NAME
    LlvmSession
  COPTS
    "-DGOOGLE_PROTOBUF_NO_RTTI"
    "-fno-rtti"
  HDRS
    "$<TARGET_PROPERTY:compiler_gym__envs__llvm__service__passes__actions_genfiles,BINARY_DIR>/ActionHeaders.h"
    "$<TARGET_PROPERTY:compiler_gym__envs__llvm__service__passes__actions_genfiles,BINARY_DIR>/ActionSwitch.h"
    "LlvmSession.h"
  SRCS
    "LlvmSession.cc"
  DEPS
    ::ActionSpace
    ::Benchmark
    ::BenchmarkFactory
    ::Cost
    ::Observation
    ::ObservationSpaces
    compiler_gym::service::CompilationSession
    compiler_gym::service::proto::compiler_gym_service_cc_grpc
    compiler_gym::third_party::autophase::InstCount
    compiler_gym::util::EnumUtil
    compiler_gym::util::GrpcStatusMacros
    compiler_gym::util::RunfilesPath
  ABS_DEPS
    Boost::filesystem
    Boost::headers
    fmt
    glog::glog
    ${_LLVM_LIBS}
    magic_enum
    nlohmann_json::nlohmann_json
    ProGraML::graph::format::node_link_graph
    ProGraML::ir::llvm::llvm-10
    ProGraML::proto::programl_cc
    CpuInfo::cpuinfo
    Clog::libclog
  INCLUDES
    ${LLVM_INCLUDE_DIRS}
    "$<TARGET_PROPERTY:compiler_gym__envs__llvm__service__passes__actions_genfiles,BINARY_DIR>"
  DEFINES
    ${LLVM_DEFINITIONS}
  PUBLIC
)

llvm_map_components_to_libnames(_LLVM_LIBS
  core support bitwriter
  )
cg_cc_library(
    NAME Observation
    SRCS Observation.cc
    HDRS Observation.h
    DEPS
      ::Benchmark
      ::Cost
      ::ObservationSpaces
      compiler_gym::service::proto::compiler_gym_service_cc_grpc
      compiler_gym::third_party::autophase::InstCount
      compiler_gym::util::GrpcStatusMacros
    ABS_DEPS
      CpuInfo::cpuinfo
      Boost::filesystem
      glog::glog
      ${_LLVM_LIBS}
      magic_enum
      nlohmann_json::nlohmann_json
      ProGraML::graph::format::node_link_graph
      ProGraML::ir::llvm::llvm-10
      ProGraML::proto::programl_cc
      Clog::libclog
  INCLUDES
    ${LLVM_INCLUDE_DIRS}
  DEFINES
    ${LLVM_DEFINITIONS}
)

cg_cc_library(
  NAME
    ObservationSpaces
  HDRS
    "ObservationSpaces.h"
  SRCS
    "ObservationSpaces.cc"
  DEPS
    ::Benchmark
    compiler_gym::service::proto::compiler_gym_service_cc
    compiler_gym::third_party::llvm::InstCount
    compiler_gym::util::EnumUtil
  ABS_DEPS
    glog::glog
    magic_enum
    nlohmann_json::nlohmann_json
    ProGraML::graph::format::node_link_graph
    ProGraML::proto::programl_cc
  PUBLIC
)
