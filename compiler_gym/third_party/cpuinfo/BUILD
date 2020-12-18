# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_foreign_cc//tools/build_defs:cmake.bzl", "cmake_external")

toolchain(
    name = "built_cmake_toolchain",
    exec_compatible_with = [
        "@bazel_tools//platforms:osx",
        "@bazel_tools//platforms:x86_64",
    ],
    toolchain = "@rules_foreign_cc//tools/build_defs/native_tools:built_cmake",
    toolchain_type = "@rules_foreign_cc//tools/build_defs:cmake_toolchain",
)

cmake_external(
    name = "cpuinfo",
    cache_entries = {
        "CPUINFO_BUILD_TOOLS": "OFF",
        "CPUINFO_BUILD_UNIT_TESTS": "OFF",
        "CPUINFO_BUILD_MOCK_TESTS": "OFF",
        "CPUINFO_BUILD_BENCHMARKS": "OFF",
    },
    lib_source = "@cpuinfo//:all",
    static_libraries = [
        "libcpuinfo.a",
        "libclog.a",
    ],
    tags = ["requires-network"],
    visibility = ["//visibility:public"],
)
