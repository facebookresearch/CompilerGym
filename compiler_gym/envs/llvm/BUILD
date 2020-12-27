# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

py_library(
    name = "llvm",
    srcs = ["__init__.py"],
    data = [
        "//compiler_gym/envs/llvm/service",
    ],
    visibility = ["//compiler_gym/envs:__pkg__"],
    deps = [
        ":benchmarks",
        ":llvm_env",
        "//compiler_gym/util",
    ],
)

py_library(
    name = "benchmarks",
    srcs = ["benchmarks.py"],
    data = [
        "//compiler_gym/third_party/llvm:clang",
        "//compiler_gym/third_party/llvm:llvm-link",
    ],
    deps = [
        "//compiler_gym/service/proto",
        "//compiler_gym/util",
    ],
)

py_library(
    name = "datasets",
    srcs = ["datasets.py"],
    deps = [
        "//compiler_gym/datasets",
    ],
)

py_library(
    name = "llvm_env",
    srcs = ["llvm_env.py"],
    data = [
        "//compiler_gym/envs/llvm/service/passes:actions_genfiles",
    ],
    deps = [
        ":datasets",
        "//compiler_gym/envs:compiler_env",
        "//compiler_gym/spaces",
        "//compiler_gym/third_party/autophase",
        "//compiler_gym/third_party/inst2vec",
    ],
)
