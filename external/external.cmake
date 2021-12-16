# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

include(ExternalProject)
include(FetchContent)
include(write_cache_script)
include(build_external_cmake_project)

unset(FETCH_CONTENT_LIST)

# # === Google test ===

set(COMPILER_GYM_GTEST_PROVIDER "internal" CACHE STRING "Find or build gtest together with Compiler Gym.")
set_property(CACHE COMPILER_GYM_GTEST_PROVIDER PROPERTY STRINGS "internal" "external")
if(COMPILER_GYM_GTEST_PROVIDER STREQUAL "internal")
  FetchContent_Declare(
      gtest
      PREFIX "${CMAKE_CURRENT_BINARY_DIR}/external/gtest"
      GIT_REPOSITORY "https://github.com/google/googletest.git"
      GIT_TAG 703bd9caab50b139428cea1aaff9974ebee5742e #tag release-1.10.0
  )
  FetchContent_MakeAvailable(gtest)
  add_library(GTest::GTest ALIAS gtest)
  add_library(GTest::Main ALIAS gtest_main)
else()
  find_package(GTest REQUIRED)
endif()

# # === Google benchmark ===

set(COMPILER_GYM_BENCHMARK_PROVIDER "internal" CACHE STRING "Find or build benchmark together with Compiler Gym.")
set_property(CACHE COMPILER_GYM_BENCHMARK_PROVIDER PROPERTY STRINGS "internal" "external")
if(COMPILER_GYM_BENCHMARK_PROVIDER STREQUAL "internal")
  FetchContent_Declare(
      benchmark
      PREFIX "${CMAKE_CURRENT_BINARY_DIR}/external/benchmark"
      GIT_REPOSITORY "https://github.com/google/benchmark.git"
      GIT_TAG 9913418d323e64a0111ca0da81388260c2bbe1e9 #tag v1.4.0
  )

  if(NOT benchmark_POPULATED)
    FetchContent_Populate(benchmark)

    # Benchmark v1.4.0 requires C++03.
    set(_CMAKE_CXX_STANDARD_OLD ${CMAKE_CXX_STANDARD})
    unset(CMAKE_CXX_STANDARD CACHE)

    option(BENCHMARK_ENABLE_TESTING "Enable testing of the benchmark library." OFF)

    add_subdirectory(${benchmark_SOURCE_DIR} ${benchmark_BINARY_DIR})

    set(CMAKE_CXX_STANDARD ${_CMAKE_CXX_STANDARD_OLD} CACHE STRING "C++ standard to be used." FORCE)
  endif()
else()
  find_package(benchmark REQUIRED)
endif()

# # === Abseil ===

set(COMPILER_GYM_ABSEIL_PROVIDER "internal" CACHE STRING "Find or build abseil together with Compiler Gym.")
set_property(CACHE COMPILER_GYM_ABSEIL_PROVIDER PROPERTY STRINGS "internal" "external")
if(COMPILER_GYM_ABSEIL_PROVIDER STREQUAL "internal")
  build_external_cmake_project(
    NAME absl
    SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/absl")
endif()
find_package(absl REQUIRED)

# # === Google flags ===

set(COMPILER_GYM_GFLAGS_PROVIDER "internal" CACHE STRING "Find or build gflags together with Compiler Gym.")
set_property(CACHE COMPILER_GYM_GFLAGS_PROVIDER PROPERTY STRINGS "internal" "external")
if(COMPILER_GYM_GFLAGS_PROVIDER STREQUAL "internal")
  build_external_cmake_project(
    NAME gflags
    SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/gflags")
endif()
find_package(gflags REQUIRED)


# # === Google logging ===

set(COMPILER_GYM_GLOG_PROVIDER "internal" CACHE STRING "Find or build glog together with Compiler Gym.")
set_property(CACHE COMPILER_GYM_GLOG_PROVIDER PROPERTY STRINGS "internal" "external")
if(COMPILER_GYM_GLOG_PROVIDER STREQUAL "internal")
  FetchContent_Declare(
      glog
      PREFIX "${CMAKE_CURRENT_BINARY_DIR}/external/glog"
      GIT_REPOSITORY "https://github.com/google/glog.git"
      GIT_TAG 96a2f23dca4cc7180821ca5f32e526314395d26a #tag v0.4.0
  )
  list(APPEND FETCH_CONTENT_LIST glog)
else()
  find_package(glog REQUIRED)
endif()

# # C++ subprocess management. https://github.com/arun11299/cpp-subprocess

set(COMPILER_GYM_SUBPROCESS_PROVIDER "internal" CACHE STRING "Find or build subprocess together with Compiler Gym.")
set_property(CACHE COMPILER_GYM_SUBPROCESS_PROVIDER PROPERTY STRINGS "internal" "external")
if(COMPILER_GYM_SUBPROCESS_PROVIDER STREQUAL "internal")
  build_external_cmake_project(
    NAME subprocess
    SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/subprocess"
  )
endif()
find_package(Subprocess REQUIRED)

# # === LLVM ===

set(COMPILER_GYM_LLVM_PROVIDER "internal" CACHE STRING "Find or build llvm together with Compiler Gym.")
set_property(CACHE COMPILER_GYM_LLVM_PROVIDER PROPERTY STRINGS "internal" "external")
build_external_cmake_project(
  NAME llvm
  SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/llvm"
  CONFIG_ARGS "-DCOMPILER_GYM_LLVM_PROVIDER=${COMPILER_GYM_LLVM_PROVIDER}")
set(LLVM_SRC_DIR "${CMAKE_CURRENT_BINARY_DIR}/external/llvm/llvm/src/llvm")
find_package(LLVM 10.0.0 EXACT REQUIRED)
# In a bunch of places in the code it is used "#include <include/llvm/...>"
list(APPEND LLVM_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/external/llvm/install")


# # === Protocol buffers ===

set(COMPILER_GYM_PROTOBUF_PROVIDER "internal" CACHE STRING "Find or build protobuf together with Compiler Gym.")
set_property(CACHE COMPILER_GYM_PROTOBUF_PROVIDER PROPERTY STRINGS "internal" "external")
if(COMPILER_GYM_PROTOBUF_PROVIDER STREQUAL "internal")
  write_cache_script("${CMAKE_CURRENT_BINARY_DIR}/external/protobuf/protobuf_initial_cache.cmake")
  execute_process(
    COMMAND "${CMAKE_COMMAND}"
    -C "${CMAKE_CURRENT_BINARY_DIR}/external/protobuf/protobuf_initial_cache.cmake"
    -S "${CMAKE_CURRENT_LIST_DIR}/protobuf"
    -B "${CMAKE_CURRENT_BINARY_DIR}/external/protobuf"
    -D "CMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/external/protobuf/install"
    COMMAND_ERROR_IS_FATAL ANY
  )
  execute_process(
    COMMAND
    "${CMAKE_COMMAND}"
    --build "${CMAKE_CURRENT_BINARY_DIR}/external/protobuf"
    COMMAND_ERROR_IS_FATAL ANY
  )
  list(PREPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_BINARY_DIR}/external/protobuf/install")
  if(NOT DEFINED Protobuf_USE_STATIC_LIBS)
    set(Protobuf_USE_STATIC_LIBS ON)
  endif()
endif()
find_package(Protobuf REQUIRED)

# # === GRPC ===

set(COMPILER_GYM_GRPC_PROVIDER "internal" CACHE STRING "Find or build gRPC together with Compiler Gym.")
set_property(CACHE COMPILER_GYM_GRPC_PROVIDER PROPERTY STRINGS "internal" "external")

set(gRPC_ABSL_PROVIDER package)
if(COMPILER_GYM_GRPC_PROVIDER STREQUAL "internal")
  if (NOT DEFINED gRPC_ABSL_PROVIDER OR gRPC_ABSL_PROVIDER STREQUAL "module")
    list(APPEND _gRPC_GIT_SUBMODULES "third_party/abseil-cpp")
  endif()

  if (NOT DEFINED gRPC_ZLIB_PROVIDER OR gRPC_ZLIB_PROVIDER STREQUAL "module")
    list(APPEND _gRPC_GIT_SUBMODULES "third_party/zlib")
  endif()

  if (NOT DEFINED gRPC_CARES_PROVIDER OR gRPC_CARES_PROVIDER STREQUAL "module")
    list(APPEND _gRPC_GIT_SUBMODULES "third_party/cares/cares")
  endif()

  if (NOT DEFINED gRPC_RE2_PROVIDER OR gRPC_RE2_PROVIDER STREQUAL "module")
    list(APPEND _gRPC_GIT_SUBMODULES "third_party/re2")
  endif()

  if (NOT DEFINED gRPC_SSL_PROVIDER OR gRPC_SSL_PROVIDER STREQUAL "module")
    list(APPEND _gRPC_GIT_SUBMODULES "third_party/boringssl-with-bazel")
  endif()

  set(gRPC_PROTOBUF_PROVIDER "package" CACHE STRING "")

  # In CMake v3.19.6 if GIT_SUBMODULES changes during reconfiguration
  # the FetchContent will not populate new submodules.
  # The PREFIX directory will have to be deleted manually.
  FetchContent_Declare(
      grpc
      PREFIX "${CMAKE_CURRENT_BINARY_DIR}/external/grpc"
      GIT_REPOSITORY "https://github.com/grpc/grpc.git"
      GIT_TAG 736e3758351ced3cd842bad3ba4e2540f01bbc48 # v1.36.0
      GIT_SUBMODULES ${_gRPC_GIT_SUBMODULES}
  )
  FetchContent_MakeAvailable(grpc)
  set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)
  #TODO(boian): remove this when GrpcStatusMacros.h uses the correct include path.
  target_include_directories(grpc++ INTERFACE "${grpc_SOURCE_DIR}")
else()
  find_package(gRPC REQUIRED)
  set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc::grpc_cpp_plugin>)
endif()

# # === C++ enum trickery ===
# # https://github.com/Neargye/magic_enum

set(COMPILER_GYM_MAGIC_ENUM_PROVIDER "internal" CACHE STRING "Find or build magic_enum together with Compiler Gym.")
set_property(CACHE COMPILER_GYM_MAGIC_ENUM_PROVIDER PROPERTY STRINGS "internal" "external")
if(COMPILER_GYM_MAGIC_ENUM_PROVIDER STREQUAL "internal")
  FetchContent_Declare(
      magic_enum
      PREFIX "${CMAKE_CURRENT_BINARY_DIR}/external/magic_enum"
      GIT_REPOSITORY "https://github.com/Neargye/magic_enum.git"
      GIT_TAG 6e932ef66dbe054e039d4dba77a41a12f9f52e0c #tag 0.7.3
  )
  list(APPEND FETCH_CONTENT_LIST magic_enum)
else()
  find_package(magic_enum REQUIRED)
endif()

# # === ctuning-programs ===
# # https://github.com/ChrisCummins/ctuning-programs

# This seems to be unused.
#ExternalProject_Add(
#    ctuning-programs
#    PREFIX "${CMAKE_BINARY_DIR}/ctuning-programs"
#    URL "https://github.com/ChrisCummins/ctuning-programs/archive/c3c126fcb400f3a14b69b152f15d15eae78ef908.tar.gz"
#    URL_HASH "SHA256=5e14a49f87c70999a082cb5cf19b780d0b56186f63356f8f994dd9ffc79ec6f3"
#    CONFIGURE_COMMAND ""
#    BUILD_COMMAND ""
#    INSTALL_COMMAND ""
#)

file(GLOB CTUNING-PROGRAMS-SRCS "ctuning-programs/**")

source_group(
    ctuning-programs-all
    FILES CTUNING-PROGRAMS-SRCS
)

source_group(
    ctuning-programs-readme
    FILES "ctuning-programs/README.md"
)

# # === cBench ===
# # https://ctuning.org/wiki/index.php/CTools:CBench

FetchContent_Declare(
    cBench
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/external/cbench"
    URL "https://dl.fbaipublicfiles.com/compiler_gym/cBench_V1.1.tar.gz"
    URL_HASH "SHA256=8908d742f5223f09f9a4d10f7e06bc805a0c1694aa70974d2aae91ab627b51e6"
    DOWNLOAD_NO_EXTRACT FALSE
)
FetchContent_MakeAvailable(cBench)
FetchContent_GetProperties(cBench SOURCE_DIR cBench_SRC_DIR)

FetchContent_Declare(
    ctuning-ai
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/external/ctuning-ai"
    URL "https://github.com/ChrisCummins/ck-mlops/archive/406738ad6d1fb2c1da9daa2c09d26fccab4e0938.tar.gz"
    URL_HASH "SHA256=a82c13733696c46b5201c614fcf7229c3a74a83ce485cab2fbf17309b7564f9c"
)
FetchContent_MakeAvailable(ctuning-ai)
FetchContent_GetProperties(ctuning-ai SOURCE_DIR ctuning_ai_SRC_DIR)

# # Datasets.

FetchContent_Declare(
  cBench_consumer_tiff_data
  PREFIX "${CMAKE_CURRENT_BINARY_DIR}/external/cBench_consumer_tiff_data"
  URL      "https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_consumer_tiff_data.tar.gz"
  URL_HASH "SHA256=779abb7b7fee8733313e462e6066c16375e9209a9f7ff692fd06c7598946939a"
)
FetchContent_MakeAvailable(cBench_consumer_tiff_data)
set(cBench_consumer_tiff_data_FILE
  "${CMAKE_CURRENT_BINARY_DIR}/external/cBench_consumer_tiff_data/src/cDatasets_V1.1_consumer_tiff_data.tar.gz")

FetchContent_Declare(
  cBench_office_data
  PREFIX "${CMAKE_CURRENT_BINARY_DIR}/external/cBench_office_data"
  URL "https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_office_data.tar.gz"
  URL_HASH "SHA256=cfa09cd37cb93aba57415033905dc6308653c7b833feba5a25067bfb62999f32"
)
FetchContent_MakeAvailable(cBench_office_data)
set(cBench_office_data_FILE
  "${CMAKE_CURRENT_BINARY_DIR}/external/cBench_office_data/src/cDatasets_V1.1_office_data.tar.gz")

FetchContent_Declare(
  cBench_telecom_data
  PREFIX "${CMAKE_CURRENT_BINARY_DIR}/external/cBench_telecom_data"
  URL "https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_telecom_data.tar.gz"
  URL_HASH "SHA256=e5cb6663beefe32fd12f90c8f533f8e1bce2f05ee4e3836efb5556d5e1089df0"
)
FetchContent_MakeAvailable(cBench_telecom_data)
set(cBench_telecom_data_FILE
  "${CMAKE_CURRENT_BINARY_DIR}/external/cBench_telecom_data/src/cDatasets_V1.1_telecom_data.tar.gz")

FetchContent_Declare(
  cBench_consumer_jpeg_data
  PREFIX "${CMAKE_CURRENT_BINARY_DIR}/external/cBench_consumer_jpeg_data"
  URL "https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_consumer_jpeg_data.tar.gz"
  URL_HASH "SHA256=bec5ffc15cd2f952d9a786f3cd31d90955c318a5e4f69c5ba472f79d5a3e8f0b"
)
FetchContent_MakeAvailable(cBench_consumer_jpeg_data)
set(cBench_consumer_jpeg_data_FILE
  "${CMAKE_CURRENT_BINARY_DIR}/external/cBench_consumer_jpeg_data/src/cDatasets_V1.1_consumer_jpeg_data.tar.gz")

FetchContent_Declare(
  cBench_telecom_gsm_data
  PREFIX "${CMAKE_CURRENT_BINARY_DIR}/external/cBench_telecom_gsm_data"
  URL "https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_telecom_gsm_data.tar.gz"
  URL_HASH "SHA256=52545d3a0ce15021131c62d96d3a3d7e6670e2d6c34226ac9a3d5191a1ee214a"
)
FetchContent_MakeAvailable(cBench_telecom_gsm_data)
set(cBench_telecom_gsm_data_FILE
  "${CMAKE_CURRENT_BINARY_DIR}/external/cBench_telecom_gsm_data/src/cDatasets_V1.1_telecom_gsm_data.tar.gz")

FetchContent_Declare(
  cBench_consumer_data
  PREFIX "${CMAKE_CURRENT_BINARY_DIR}/external/cBench_consumer_data"
  URL "https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_consumer_data.tar.gz"
  URL_HASH "SHA256=a4d40344af3022bfd7b4c6fcf6d59d598825b07d9e37769dbf1b3effa39aa445"
)
FetchContent_MakeAvailable(cBench_consumer_data)
set(cBench_consumer_data_FILE
  "${CMAKE_CURRENT_BINARY_DIR}/external/cBench_consumer_data/src/cDatasets_V1.1_consumer_data.tar.gz")

FetchContent_Declare(
  cBench_bzip2_data
  PREFIX "${CMAKE_CURRENT_BINARY_DIR}/external/cBench_bzip2_data"
  URL "https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_bzip2_data.tar.gz"
  URL_HASH "SHA256=46e5760eeef77e6b0c273af92de971bc45f33a59e0efc183073d9aa6b716c302"
)
FetchContent_MakeAvailable(cBench_bzip2_data)
set(cBench_bzip2_data_FILE
  "${CMAKE_CURRENT_BINARY_DIR}/external/cBench_bzip2_data/src/cDatasets_V1.1_bzip2_data.tar.gz")

FetchContent_Declare(
  cBench_network_patricia_data
  PREFIX "${CMAKE_CURRENT_BINARY_DIR}/external/cBench_network_patricia_data"
  URL "https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_network_patricia_data.tar.gz"
  URL_HASH "SHA256=72dae0e670d93ef929e50aca7a138463e0915502281ccafe793e378cb2a85dfb"
)
FetchContent_MakeAvailable(cBench_network_patricia_data)
set(cBench_network_patricia_data_FILE
  "${CMAKE_CURRENT_BINARY_DIR}/external/cBench_network_patricia_data/src/cDatasets_V1.1_network_patricia_data.tar.gz")

FetchContent_Declare(
  cBench_network_dijkstra_data
  PREFIX "${CMAKE_CURRENT_BINARY_DIR}/external/cBench_network_dijkstra_data"
  URL "https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_network_dijkstra_data.tar.gz"
  URL_HASH "SHA256=41c13f59cdfbc772081cd941f499b030370bc570fc2ba60a5c4b7194bc36ca5f"
)
FetchContent_MakeAvailable(cBench_network_dijkstra_data)
set(cBench_network_dijkstra_data_FILE
  "${CMAKE_CURRENT_BINARY_DIR}/external/cBench_network_dijkstra_data/src/cDatasets_V1.1_network_dijkstra_data.tar.gz")

FetchContent_Declare(
  cBench_automotive_susan_data
  PREFIX "${CMAKE_CURRENT_BINARY_DIR}/external/cBench_automotive_susan_data"
  URL "https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_automotive_susan_data.tar.gz"
  URL_HASH "SHA256=df56e1e44ccc560072381cdb001d770003ac74f92593dd5dbdfdd4ff9332a8e6"
)
FetchContent_MakeAvailable(cBench_automotive_susan_data)
set(cBench_automotive_susan_data_FILE
  "${CMAKE_CURRENT_BINARY_DIR}/external/cBench_automotive_susan_data/src/cDatasets_V1.1_automotive_susan_data.tar.gz")

FetchContent_Declare(
  cBench_automotive_qsort_data
  PREFIX "${CMAKE_CURRENT_BINARY_DIR}/external/cBench_automotive_qsort_data"
  URL "https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_automotive_qsort_data.tar.gz"
  URL_HASH "SHA256=510b4225021408ac190f6f793e7d7171d3553c9916cfa8b2fb4ace005105e768"
)
FetchContent_MakeAvailable(cBench_automotive_qsort_data)
set(cBench_automotive_qsort_data_FILE
  "${CMAKE_CURRENT_BINARY_DIR}/external/cBench_automotive_qsort_data/src/cDatasets_V1.1_automotive_qsort_data.tar.gz")

# # === C++ cpuinfo ===

set(COMPILER_GYM_CPUINFO_PROVIDER "internal" CACHE STRING "Find or build cpuinfo together with Compiler Gym.")
set_property(CACHE COMPILER_GYM_CPUINFO_PROVIDER PROPERTY STRINGS "internal" "external")
if(COMPILER_GYM_CPUINFO_PROVIDER STREQUAL "internal")
  build_external_cmake_project(
    NAME cpuinfo
    SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/cpuinfo")
endif()
set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH TRUE)
find_package(PkgConfig REQUIRED)
pkg_check_modules(CpuInfo REQUIRED IMPORTED_TARGET libcpuinfo)
add_library(CpuInfo::cpuinfo ALIAS PkgConfig::CpuInfo)

find_package(Clog REQUIRED)
# For some reason this does not propagate to the linker when CpuInfo::cpuinfo is included
#get_target_property(_CpuInfo_LINK_LIBS PkgConfig::CpuInfo IMPORTED_LINK_INTERFACE_LIBRARIES)
#if (NOT _CpuInfo_LINK_LIBS)
#  set(_CpuInfo_LINK_LIBS Clog::libclog)
#else()
#  list(APPEND _CpuInfo_LINK_LIBS Clog::libclog)
#endif()
#set_target_properties(PkgConfig::CpuInfo
#  PROPERTIES IMPORTED_LINK_INTERFACE_LIBRARIES
#  "${_CpuInfo_LINK_LIBS}")


# # === Csmith ===
# # https://embed.cs.utah.edu/csmith/

build_external_cmake_project(
  NAME csmith
  SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/csmith"
  INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/external/csmith/install/csmith")
find_package(Csmith REQUIRED)

# # === DeepDataFlow ===
# # https://zenodo.org/record/4122437

#FetchContent_Declare(
#    DeepDataFlow
#    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/external/DeepDataFlow"
#    SOURCE_DIR "${CMAKE_BINARY_DIR}/compiler_gym/third_party/DeepDataFlow"
#    URL "https://zenodo.org/record/4122437/files/llvm_bc_20.06.01.tar.bz2?download=1"
#    URL_HASH "SHA256=ea6accbeb005889db3ecaae99403933c1008e0f2f4adc3c4afae3d7665c54004"
#)
#list(APPEND FETCH_CONTENT_LIST DeepDataFlow)

# === A modern C++ formatting library ===
# https://fmt.dev

set(COMPILER_GYM_FMT_PROVIDER "internal" CACHE STRING "Find or build fmt together with Compiler Gym.")
set_property(CACHE COMPILER_GYM_FMT_PROVIDER PROPERTY STRINGS "internal" "external")
if(COMPILER_GYM_FMT_PROVIDER STREQUAL "internal")
  FetchContent_Declare(
      fmt
      PREFIX "${CMAKE_CURRENT_BINARY_DIR}/external/fmt"
      GIT_REPOSITORY "https://github.com/fmtlib/fmt.git"
      GIT_TAG f94b7364b9409f05207c3af3fa4666730e11a854 #tag 6.1.2
  )
  FetchContent_MakeAvailable(fmt)
else()
  find_package(fmt REQUIRED)
endif()

# # === Boost ===

set(COMPILER_GYM_BOOST_PROVIDER "internal" CACHE STRING "Find or build boost together with Compiler Gym.")
set_property(CACHE COMPILER_GYM_BOOST_PROVIDER PROPERTY STRINGS "internal" "external")
if(COMPILER_GYM_BOOST_PROVIDER STREQUAL "internal")
  build_external_cmake_project(
    NAME boost
    SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/boost")
    if (NOT DEFINED Boost_USE_STATIC_LIBS)
      set(Boost_USE_STATIC_LIBS ON)
    endif()
endif()
find_package(Boost REQUIRED COMPONENTS filesystem headers)

# # === nlohmann_json ===

set(COMPILER_GYM_NLOHMANN_JSON_PROVIDER "internal" CACHE STRING "Find or build nlohmann_json together with Compiler Gym.")
set_property(CACHE COMPILER_GYM_NLOHMANN_JSON_PROVIDER PROPERTY STRINGS "internal" "external")
if(COMPILER_GYM_NLOHMANN_JSON_PROVIDER STREQUAL "internal")
  FetchContent_Declare(
      nlohmann_json
      PREFIX "${CMAKE_CURRENT_BINARY_DIR}/external/nlohmann_json"
      GIT_REPOSITORY "https://github.com/nlohmann/json.git"
      GIT_TAG e7b3b40b5a95bc74b9a7f662830a27c49ffc01b4 #tag: v3.7.3
  )
  list(APPEND FETCH_CONTENT_LIST nlohmann_json)
else()
  find_package(nlohmann_json REQUIRED)
endif()

# # === ProGraML ===
# # https://github.com/ChrisCummins/ProGraML

build_external_cmake_project(
  NAME programl
  SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/programl")
list(PREPEND CMAKE_PREFIX_PATH
  "${CMAKE_CURRENT_BINARY_DIR}/external/programl/programl/src/programl/bazel-bin"
  "${CMAKE_CURRENT_BINARY_DIR}/external/programl/programl/src/programl/bazel-bin/external/labm8"
  "${CMAKE_CURRENT_BINARY_DIR}/external/programl/programl/src/programl/bazel-programl"
  "${CMAKE_CURRENT_BINARY_DIR}/external/programl/programl/src/programl/bazel-programl/external/labm8"
  )
find_package(Labm8 REQUIRED)
find_package(ProGraML REQUIRED)

FetchContent_MakeAvailable(${FETCH_CONTENT_LIST})
