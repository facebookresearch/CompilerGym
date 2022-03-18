# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

set(_DEPS "compiler_gym-mlir-service")
if(DARWIN)
    #TODO(boian): figure out what is this target.
    #list(APPEND _DEPS "@llvm//:darwin")
endif()
cg_filegroup(
  NAME "service"
  DEPENDS ${_DEPS}
)

cg_cc_binary(
  NAME
    compiler_gym-mlir-service
  SRCS
    "RunService.cc"
  DEPS
    ::MlirSession
    compiler_gym::service::runtime::cc_runtime
  ABS_DEPS
    LLVMSupport
    MLIRIR
    MLIRSupport
    MLIROptLib
    MLIRPass

    MLIRAffine
    MLIRAffineTransforms
    MLIRGPUOps
    MLIRGPUToSPIRV
    MLIRIR
    MLIRLLVMIR
    MLIRLinalg
    MLIRLinalgToLLVM
    MLIRLinalgToSPIRV
    MLIRLinalgTransforms
    MLIRQuant
    MLIRSCF
    MLIRSCFToGPU
    MLIRSCFTransforms
    MLIRSPIRV
    MLIRSPIRVTransforms
    MLIRShape
    MLIRStandard
    MLIRStandardToSPIRV
    MLIRTosa
    MLIRTosaTransforms
    MLIRTransforms
    MLIRArmNeon
    MLIRArmNeon2dToIntr
    MLIRVector

    MLIREmitC

    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRTargetLLVMIRExport
)

cg_cc_library(
  NAME
    ActionSpace
  HDRS
    "ActionSpace.h"
  SRCS
    "ActionSpace.cc"
  DEPS
    compiler_gym::service::proto::compiler_gym_service_cc
    compiler_gym::util::EnumUtil
    compiler_gym::util::Unreachable
  ABS_DEPS
    Boost::headers
    fmt
    magic_enum
  PUBLIC
)

message(STATUS "MLIR_INCLUDE_DIRS: ${MLIR_INCLUDE_DIRS}")

llvm_map_components_to_libnames(_LLVM_LIBS core support bitwriter)
# list(APPEND _LLVM_LIBS MLIRIR MLIRAffine)
cg_cc_library(
  NAME
    Benchmark
  HDRS
    "Benchmark.h"
  SRCS
    "Benchmark.cc"
  DEPS
    ::MlirUtils
    compiler_gym::service::proto::compiler_gym_service_cc
    compiler_gym::util::GrpcStatusMacros
    compiler_gym::util::RunfilesPath
    compiler_gym::util::Subprocess
  ABS_DEPS
    MLIRSupport
    MLIRIR
    MLIRPass
    Boost::filesystem
    grpc++
    fmt
    glog::glog
    nlohmann_json::nlohmann_json
    ${_LLVM_LIBS}
  INCLUDES
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
  DEFINES
    ${LLVM_DEFINITIONS}
  PUBLIC
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
    ::MlirUtils
    compiler_gym::service::proto::compiler_gym_service_cc
    compiler_gym::util::GrpcStatusMacros
    compiler_gym::util::RunfilesPath
    compiler_gym::util::StrLenConstexpr
  ABS_DEPS
    MLIRSupport
    MLIRIR
    MLIRPass
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

llvm_map_components_to_libnames(_LLVM_LIBS
  core analysis coroutines objcarcopts target codegen
  x86codegen x86asmparser #TODO(boian): can these be found programmatically
)
#message(FATAL_ERROR "_LLVM_LIBS=${_LLVM_LIBS}")
cg_cc_library(
  NAME
    MlirSession
  COPTS
    "-DGOOGLE_PROTOBUF_NO_RTTI"
    "-fno-rtti"
  HDRS
    "MlirSession.h"
  SRCS
    "MlirSession.cc"
  DEPS
    ::ActionSpace
    ::Benchmark
    ::BenchmarkFactory
    ::MlirUtils
    ::Observation
    ::ObservationSpaces
    compiler_gym::service::CompilationSession
    compiler_gym::service::proto::compiler_gym_service_cc_grpc
    compiler_gym::service::proto::proto_cc
    compiler_gym::util::EnumUtil
    compiler_gym::util::GrpcStatusMacros
    compiler_gym::util::RunfilesPath
  ABS_DEPS
    MLIRSupport
    MLIRIR
    MLIRPass
    Boost::filesystem
    Boost::headers
    fmt
    glog::glog
    ${_LLVM_LIBS}
    magic_enum
    nlohmann_json::nlohmann_json
    CpuInfo::cpuinfo
    Clog::libclog
  INCLUDES
    ${LLVM_INCLUDE_DIRS}
  DEFINES
    ${LLVM_DEFINITIONS}
  PUBLIC
)

llvm_map_components_to_libnames(_LLVM_LIBS
  core analysis coroutines objcarcopts target codegen support bitwriter
  x86codegen x86asmparser #TODO(boian): can these be found programmatically
)
cg_cc_library(
  NAME
    MlirUtils
  COPTS
    "-DGOOGLE_PROTOBUF_NO_RTTI"
    "-fno-rtti"
  HDRS
    "MlirUtils.h"
  SRCS
    "MlirUtils.cc"
  DEPS
    compiler_gym::service::proto::compiler_gym_service_cc_grpc
    compiler_gym::util::GrpcStatusMacros
  ABS_DEPS
    fmt
    glog::glog
    ${_LLVM_LIBS}
    MLIRAffineToStandard
    MLIRAsync
    MLIRDLTI
    MLIREmitC
    MLIRGPUOps
    MLIRIR
    MLIRLinalg
    MLIRLinalgToLLVM
    MLIRLinalgTransforms
    MLIRLinalgUtils
    MLIRLLVMToLLVMIRTranslation
    MLIRMemRefToLLVM
    MLIRNVVMIR
    MLIROpenACC
    MLIROpenMP
    MLIRPass
    MLIRQuant
    MLIRROCDLIR
    MLIRSCFToStandard
    MLIRShape
    MLIRSparseTensor
    MLIRSPIRV
    MLIRStandardToLLVM
    MLIRSupport
    MLIRTensor
    MLIRTensorInferTypeOpInterfaceImpl
    MLIRTensorTilingInterfaceImpl
    MLIRTosa
    MLIRVectorToLLVM
    MLIRVectorToSCF
  INCLUDES
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
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
      ::ObservationSpaces
      compiler_gym::service::proto::compiler_gym_service_cc_grpc
      compiler_gym::util::GrpcStatusMacros
    ABS_DEPS
      CpuInfo::cpuinfo
      Boost::filesystem
      glog::glog
      ${_LLVM_LIBS}
      magic_enum
      nlohmann_json::nlohmann_json
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
    compiler_gym::util::EnumUtil
  ABS_DEPS
    glog::glog
    magic_enum
    nlohmann_json::nlohmann_json
  PUBLIC
)
