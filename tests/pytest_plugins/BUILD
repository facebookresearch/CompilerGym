# Plugins and test fixtures for CompilerGym.
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
package(default_visibility = ["//visibility:public"])

py_library(
    name = "llvm",
    testonly = 1,
    srcs = ["llvm.py"],
    data = [
        "//compiler_gym/envs/llvm/service/passes:actions_genfiles",
        "//compiler_gym/third_party/cBench:benchmarks_list",
    ],
    deps = [
        "//compiler_gym",
        "//compiler_gym/envs",
        "//compiler_gym/envs/llvm:datasets",
        "//compiler_gym/util",
        "//tests:test_main",
    ],
)

py_library(
    name = "common",
    testonly = 1,
    srcs = ["common.py"],
)
