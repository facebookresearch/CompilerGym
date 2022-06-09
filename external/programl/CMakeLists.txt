# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

project(gflags)
cmake_minimum_required(VERSION 3.15)

include(ExternalProject)

list(
    APPEND
    CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_LIST_DIR}/../../build_tools/cmake"
)
find_package(Bazel REQUIRED)

externalproject_add(
    programl
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/programl"
    URL
        "https://github.com/ChrisCummins/ProGraML/archive/83f00233b04f4ecf7f12a79c80ffe23c2953913f.tar.gz"
    URL_HASH
        "SHA256=704826311b842b3ffc2dedcce593fdd319d76e95bdf5386985768007ebb22316"
    DOWNLOAD_NO_EXTRACT FALSE
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

externalproject_add_step(
    programl
    build_programl
    ALWAYS TRUE
    COMMAND
        timeout 10s "${CMAKE_COMMAND}" -E env "CC=${CMAKE_C_COMPILER}"
        "CXX=${CMAKE_CXX_COMPILER}" "${Bazel_EXECUTABLE}" build
        --verbose_failures "--cxxopt=-std=c++${CMAKE_CXX_STANDARD}" --
        //programl/graph:features //programl/graph:program_graph_builder
        //programl/graph/format:node_link_graph //programl/ir/llvm:llvm-10
        //programl/proto:programl_cc //programl/proto:programl
        @labm8//labm8/cpp:logging @labm8//labm8/cpp:status
        @labm8//labm8/cpp:status_macros @labm8//labm8/cpp:statusor
        @labm8//labm8/cpp:string @labm8//labm8/cpp:stringpiece
    DEPENDEES update
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/programl/src/programl"
    USES_TERMINAL TRUE
)
