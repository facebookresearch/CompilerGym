# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

project(benchmark)
cmake_minimum_required(VERSION 3.15)

include(FetchContent)

fetchcontent_declare(
    benchmark
    PREFIX
    "${CMAKE_CURRENT_BINARY_DIR}/external/benchmark"
    GIT_REPOSITORY "https://github.com/google/benchmark.git"
    GIT_TAG
        0d98dba29d66e93259db7daa53a9327df767a415 #tag v1.6.1
)

if(NOT benchmark_POPULATED)
    fetchcontent_populate(benchmark)

    # Benchmark v1.4.0 requires C++03.
    set(_CMAKE_CXX_STANDARD_OLD ${CMAKE_CXX_STANDARD})
    unset(CMAKE_CXX_STANDARD CACHE)

    option(
        BENCHMARK_ENABLE_TESTING
        "Enable testing of the benchmark library."
        OFF
    )

    add_subdirectory(${benchmark_SOURCE_DIR} ${benchmark_BINARY_DIR})

    set(CMAKE_CXX_STANDARD ${_CMAKE_CXX_STANDARD_OLD}
        CACHE STRING "C++ standard to be used."
        FORCE
    )
endif()
