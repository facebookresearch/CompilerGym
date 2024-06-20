# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

include(ExternalProject)
include(FetchContent)
include(write_cache_script)
include(build_external_cmake_project)

unset(FETCH_CONTENT_LIST)

# === Google test ===

set(COMPILER_GYM_GTEST_PROVIDER "internal"
    CACHE STRING "Find or build gtest together with Compiler Gym."
)
set_property(
    CACHE COMPILER_GYM_GTEST_PROVIDER
    PROPERTY STRINGS "internal" "external"
)
if(COMPILER_GYM_GTEST_PROVIDER STREQUAL "internal")
    fetchcontent_declare(
        gtest
        PREFIX
        "${CMAKE_CURRENT_BINARY_DIR}/external/gtest"
        GIT_REPOSITORY "https://github.com/google/googletest.git"
        GIT_TAG
            703bd9caab50b139428cea1aaff9974ebee5742e #tag release-1.10.0
    )
    fetchcontent_makeavailable(gtest)
    add_library(GTest::GTest ALIAS gtest)
    add_library(GTest::Main ALIAS gtest_main)
else()
    find_package(GTest REQUIRED)
endif()

# === Google benchmark ===

set(COMPILER_GYM_BENCHMARK_PROVIDER "internal"
    CACHE STRING "Find or build benchmark together with Compiler Gym."
)
set_property(
    CACHE COMPILER_GYM_BENCHMARK_PROVIDER
    PROPERTY STRINGS "internal" "external"
)
if(COMPILER_GYM_BENCHMARK_PROVIDER STREQUAL "internal")
    build_external_cmake_project(
      NAME benchmark
      SRC_DIR   "${CMAKE_CURRENT_LIST_DIR}/benchmark"
    )
endif()
find_package(benchmark REQUIRED)

# === Abseil ===

set(COMPILER_GYM_ABSEIL_PROVIDER "internal"
    CACHE STRING "Find or build abseil together with Compiler Gym."
)
set_property(
    CACHE COMPILER_GYM_ABSEIL_PROVIDER
    PROPERTY STRINGS "internal" "external"
)
if(COMPILER_GYM_ABSEIL_PROVIDER STREQUAL "internal")
    build_external_cmake_project(
      NAME absl
      SRC_DIR   "${CMAKE_CURRENT_LIST_DIR}/absl"
    )
endif()
find_package(absl REQUIRED)

# === Google flags ===

set(COMPILER_GYM_GFLAGS_PROVIDER "internal"
    CACHE STRING "Find or build gflags together with Compiler Gym."
)
set_property(
    CACHE COMPILER_GYM_GFLAGS_PROVIDER
    PROPERTY STRINGS "internal" "external"
)
if(COMPILER_GYM_GFLAGS_PROVIDER STREQUAL "internal")
    build_external_cmake_project(
      NAME gflags
      SRC_DIR   "${CMAKE_CURRENT_LIST_DIR}/gflags"
    )
endif()
find_package(gflags REQUIRED)

# === Google logging ===

set(COMPILER_GYM_GLOG_PROVIDER "internal"
    CACHE STRING "Find or build glog together with Compiler Gym."
)
set_property(
    CACHE COMPILER_GYM_GLOG_PROVIDER
    PROPERTY STRINGS "internal" "external"
)
if(COMPILER_GYM_GLOG_PROVIDER STREQUAL "internal")
    fetchcontent_declare(
        glog
        PREFIX
        "${CMAKE_CURRENT_BINARY_DIR}/external/glog"
        GIT_REPOSITORY "https://github.com/google/glog.git"
        GIT_TAG
            96a2f23dca4cc7180821ca5f32e526314395d26a #tag v0.4.0
    )
    list(APPEND FETCH_CONTENT_LIST glog)
else()
    find_package(glog REQUIRED)
endif()

# === IR2Vec ===
# https://github.com/IITH-Compilers/IR2Vec

set(COMPILER_GYM_IR2VEC_PROVIDER "internal"
    CACHE STRING "Find or build IR2Vec together with Compiler Gym."
)
set_property(
    CACHE COMPILER_GYM_IR2VEC_PROVIDER
    PROPERTY STRINGS "internal" "external"
)
if(COMPILER_GYM_IR2VEC_PROVIDER STREQUAL "internal")
    build_external_cmake_project(
      NAME ir2vec
      SRC_DIR     "${CMAKE_CURRENT_LIST_DIR}/ir2vec"
    )
else()
    find_package(ir2vec REQUIRED)
endif()

# === LLVM ===

set(COMPILER_GYM_LLVM_PROVIDER "internal"
    CACHE STRING "Find or build llvm together with Compiler Gym."
)
set_property(
    CACHE COMPILER_GYM_LLVM_PROVIDER
    PROPERTY STRINGS "internal" "external"
)

# === LLVM 10.0.0 ===

if(COMPILER_GYM_ENABLE_LLVM_ENV)
    if(COMPILER_GYM_LLVM_PROVIDER STREQUAL "internal")
        build_external_cmake_project(
          NAME llvm
          SRC_DIR   "${CMAKE_CURRENT_LIST_DIR}/llvm"
          CONFIG_ARGS   "-DCOMPILER_GYM_LLVM_PROVIDER=${COMPILER_GYM_LLVM_PROVIDER}"
        )
    endif()
    find_package(LLVM 10.0.0 EXACT REQUIRED)
    message("Using LLVM version ${LLVM_VERSION} from ${LLVM_DIR}")
endif()

# === LLVM 14 ===

if(COMPILER_GYM_ENABLE_MLIR_ENV)
    if(COMPILER_GYM_LLVM_PROVIDER STREQUAL "internal")
        build_external_cmake_project(
            NAME llvm-14
            SRC_DIR   "${CMAKE_CURRENT_LIST_DIR}/llvm-14"
            CONFIG_ARGS   "-DCOMPILER_GYM_LLVM_PROVIDER=${COMPILER_GYM_LLVM_PROVIDER}"
        )
        find_package(LLVM 14.0.0 EXACT REQUIRED)
    else()
        find_package(LLVM REQUIRED)
    endif()
    find_package(Clang REQUIRED CONFIG)
    find_package(MLIR REQUIRED CONFIG)
    message("Using LLVM version ${LLVM_VERSION} from ${LLVM_DIR}")
endif()

# === Protocol buffers ===
set(COMPILER_GYM_PROTOBUF_PROVIDER "internal"
    CACHE STRING "Find or build protobuf together with Compiler Gym."
)
set_property(
    CACHE COMPILER_GYM_PROTOBUF_PROVIDER
    PROPERTY STRINGS "internal" "external"
)
if(COMPILER_GYM_PROTOBUF_PROVIDER STREQUAL "internal")
    build_external_cmake_project(
      NAME protobuf
      SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/protobuf"
      NO_INSTALL
    )
    if(NOT DEFINED Protobuf_USE_STATIC_LIBS)
        set(Protobuf_USE_STATIC_LIBS ON)
    endif()
endif()
find_package(Protobuf REQUIRED)

# === GRPC ===

set(COMPILER_GYM_GRPC_PROVIDER "internal"
    CACHE STRING "Find or build gRPC together with Compiler Gym."
)
set_property(
    CACHE COMPILER_GYM_GRPC_PROVIDER
    PROPERTY STRINGS "internal" "external"
)

set(gRPC_ABSL_PROVIDER package)
if(COMPILER_GYM_GRPC_PROVIDER STREQUAL "internal")
    if(NOT DEFINED gRPC_ABSL_PROVIDER OR gRPC_ABSL_PROVIDER STREQUAL "module")
        list(APPEND _gRPC_GIT_SUBMODULES "third_party/abseil-cpp")
    endif()

    if(NOT DEFINED gRPC_ZLIB_PROVIDER OR gRPC_ZLIB_PROVIDER STREQUAL "module")
        list(APPEND _gRPC_GIT_SUBMODULES "third_party/zlib")
    endif()

    if(NOT DEFINED gRPC_CARES_PROVIDER OR gRPC_CARES_PROVIDER STREQUAL "module")
        list(APPEND _gRPC_GIT_SUBMODULES "third_party/cares/cares")
    endif()

    if(NOT DEFINED gRPC_RE2_PROVIDER OR gRPC_RE2_PROVIDER STREQUAL "module")
        list(APPEND _gRPC_GIT_SUBMODULES "third_party/re2")
    endif()

    if(NOT DEFINED gRPC_SSL_PROVIDER OR gRPC_SSL_PROVIDER STREQUAL "module")
        list(APPEND _gRPC_GIT_SUBMODULES "third_party/boringssl-with-bazel")
    endif()

    set(gRPC_PROTOBUF_PROVIDER "package" CACHE STRING "")

    # In CMake v3.19.6 if GIT_SUBMODULES changes during reconfiguration
    # the FetchContent will not populate new submodules.
    # The PREFIX directory will have to be deleted manually.
    fetchcontent_declare(
        grpc
        PREFIX
        "${CMAKE_CURRENT_BINARY_DIR}/external/grpc"
        GIT_REPOSITORY "https://github.com/grpc/grpc.git"
        GIT_TAG
            0e34ec3396726ba44a7aaf752eccac0e998beaa8 # v1.49.2
        GIT_SUBMODULES ${_gRPC_GIT_SUBMODULES}
    )
    fetchcontent_makeavailable(grpc)
    set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)
else()
    find_package(gRPC REQUIRED)
    set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc::grpc_cpp_plugin>)
endif()

# === C++ enum trickery ===
# https://github.com/Neargye/magic_enum

set(COMPILER_GYM_MAGIC_ENUM_PROVIDER "internal"
    CACHE STRING "Find or build magic_enum together with Compiler Gym."
)
set_property(
    CACHE COMPILER_GYM_MAGIC_ENUM_PROVIDER
    PROPERTY STRINGS "internal" "external"
)
if(COMPILER_GYM_MAGIC_ENUM_PROVIDER STREQUAL "internal")
    fetchcontent_declare(
        magic_enum
        PREFIX
        "${CMAKE_CURRENT_BINARY_DIR}/external/magic_enum"
        GIT_REPOSITORY "https://github.com/Neargye/magic_enum.git"
        GIT_TAG
            6e932ef66dbe054e039d4dba77a41a12f9f52e0c #tag 0.7.3
    )
    list(APPEND FETCH_CONTENT_LIST magic_enum)
else()
    find_package(magic_enum REQUIRED)
endif()

# === ctuning-programs ===
# https://github.com/ChrisCummins/ctuning-programs

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

source_group(ctuning-programs-all FILES CTUNING-PROGRAMS-SRCS)

source_group(ctuning-programs-readme FILES "ctuning-programs/README.md")

# === cBench ===
# https://ctuning.org/wiki/index.php/CTools:CBench

fetchcontent_declare(
    cBench
    PREFIX
    "${CMAKE_CURRENT_BINARY_DIR}/external/cbench"
    URL "https://dl.fbaipublicfiles.com/compiler_gym/cBench_V1.1.tar.gz"
    URL_HASH
        "SHA256=8908d742f5223f09f9a4d10f7e06bc805a0c1694aa70974d2aae91ab627b51e6"
    DOWNLOAD_NO_EXTRACT FALSE
)
fetchcontent_makeavailable(cBench)
fetchcontent_getproperties(cBench SOURCE_DIR cBench_SRC_DIR)

fetchcontent_declare(
    ctuning-ai
    PREFIX
    "${CMAKE_CURRENT_BINARY_DIR}/external/ctuning-ai"
    URL
        "https://github.com/ChrisCummins/ck-mlops/archive/406738ad6d1fb2c1da9daa2c09d26fccab4e0938.tar.gz"
    URL_HASH
        "SHA256=a82c13733696c46b5201c614fcf7229c3a74a83ce485cab2fbf17309b7564f9c"
)
fetchcontent_makeavailable(ctuning-ai)
fetchcontent_getproperties(ctuning-ai SOURCE_DIR ctuning_ai_SRC_DIR)

# === C++ cpuinfo ===

set(COMPILER_GYM_CPUINFO_PROVIDER "internal"
    CACHE STRING "Find or build cpuinfo together with Compiler Gym."
)
set_property(
    CACHE COMPILER_GYM_CPUINFO_PROVIDER
    PROPERTY STRINGS "internal" "external"
)
if(COMPILER_GYM_CPUINFO_PROVIDER STREQUAL "internal")
    build_external_cmake_project(
      NAME cpuinfo
      SRC_DIR   "${CMAKE_CURRENT_LIST_DIR}/cpuinfo"
    )
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

# === Csmith ===
# https://embed.cs.utah.edu/csmith/

build_external_cmake_project(
  NAME csmith
  SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/csmith"
  INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/external/csmith/install/csmith"
)
find_package(Csmith REQUIRED)

# === DeepDataFlow ===
# https://zenodo.org/record/4122437

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

set(COMPILER_GYM_FMT_PROVIDER "internal"
    CACHE STRING "Find or build fmt together with Compiler Gym."
)
set_property(
    CACHE COMPILER_GYM_FMT_PROVIDER
    PROPERTY STRINGS "internal" "external"
)
if(COMPILER_GYM_FMT_PROVIDER STREQUAL "internal")
    fetchcontent_declare(
        fmt
        PREFIX
        "${CMAKE_CURRENT_BINARY_DIR}/external/fmt"
        GIT_REPOSITORY "https://github.com/fmtlib/fmt.git"
        GIT_TAG
            f94b7364b9409f05207c3af3fa4666730e11a854 #tag 6.1.2
    )
    fetchcontent_makeavailable(fmt)
else()
    find_package(fmt REQUIRED)
endif()

# === Boost ===

set(COMPILER_GYM_BOOST_PROVIDER "internal"
    CACHE STRING "Find or build boost together with Compiler Gym."
)
set_property(
    CACHE COMPILER_GYM_BOOST_PROVIDER
    PROPERTY STRINGS "internal" "external"
)
if(COMPILER_GYM_BOOST_PROVIDER STREQUAL "internal")
    build_external_cmake_project(
      NAME boost
      SRC_DIR   "${CMAKE_CURRENT_LIST_DIR}/boost"
    )
    if(NOT DEFINED Boost_USE_STATIC_LIBS)
        set(Boost_USE_STATIC_LIBS ON)
    endif()
endif()
find_package(Boost REQUIRED COMPONENTS filesystem headers)

# === nlohmann_json ===

set(COMPILER_GYM_NLOHMANN_JSON_PROVIDER "internal"
    CACHE STRING "Find or build nlohmann_json together with Compiler Gym."
)
set_property(
    CACHE COMPILER_GYM_NLOHMANN_JSON_PROVIDER
    PROPERTY STRINGS "internal" "external"
)
if(COMPILER_GYM_NLOHMANN_JSON_PROVIDER STREQUAL "internal")
    fetchcontent_declare(
        nlohmann_json
        PREFIX
        "${CMAKE_CURRENT_BINARY_DIR}/external/nlohmann_json"
        GIT_REPOSITORY "https://github.com/nlohmann/json.git"
        GIT_TAG
            e7b3b40b5a95bc74b9a7f662830a27c49ffc01b4 #tag: v3.7.3
    )
    list(APPEND FETCH_CONTENT_LIST nlohmann_json)
else()
    find_package(nlohmann_json REQUIRED)
endif()

# === ProGraML ===
# https://github.com/ChrisCummins/ProGraML

if(COMPILER_GYM_ENABLE_LLVM_ENV)
    build_external_cmake_project(
      NAME programl
      SRC_DIR   "${CMAKE_CURRENT_LIST_DIR}/programl"
    )
    list(
        PREPEND
        CMAKE_PREFIX_PATH
        "${CMAKE_CURRENT_BINARY_DIR}/external/programl/programl/src/programl/bazel-bin"
        "${CMAKE_CURRENT_BINARY_DIR}/external/programl/programl/src/programl/bazel-bin/external/labm8"
        "${CMAKE_CURRENT_BINARY_DIR}/external/programl/programl/src/programl/bazel-programl"
        "${CMAKE_CURRENT_BINARY_DIR}/external/programl/programl/src/programl/bazel-programl/external/labm8"
    )
    find_package(Labm8 REQUIRED)
    find_package(ProGraML REQUIRED)
endif()

# === Eigen ===
# https://eigen.tuxfamily.org/index.php?title=Main_Page

set(COMPILER_GYM_EIGEN_PROVIDER "internal"
    CACHE STRING "Find or build eigen together with Compiler Gym."
)
set_property(
    CACHE COMPILER_GYM_EIGEN_PROVIDER
    PROPERTY STRINGS "internal" "external"
)
if(COMPILER_GYM_EIGEN_PROVIDER STREQUAL "internal")
    fetchcontent_declare(
        eigen
        PREFIX
        "${CMAKE_CURRENT_BINARY_DIR}/external/eigen"
        GIT_REPOSITORY "https://gitlab.com/libeigen/eigen.git"
        GIT_TAG
            21ae2afd4edaa1b69782c67a54182d34efe43f9c #tag: v3.3.7
    )
    list(APPEND FETCH_CONTENT_LIST eigen)
else()
    find_package(eigen REQUIRED)
endif()

fetchcontent_makeavailable(${FETCH_CONTENT_LIST})
