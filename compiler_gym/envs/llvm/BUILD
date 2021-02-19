# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

py_library(
    name = "llvm",
    srcs = [
        "__init__.py",
        ":specs.py",
    ],
    data = [
        "//compiler_gym/envs/llvm/service:compiler_gym-llvm-service",
    ],
    visibility = ["//compiler_gym:__subpackages__"],
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
    visibility = ["//compiler_gym:__subpackages__"],
    deps = [
        "//compiler_gym/service/proto",
        "//compiler_gym/util",
    ],
)

py_library(
    name = "datasets",
    srcs = ["datasets.py"],
    data = [
        "//compiler_gym/third_party/llvm:clang",
    ],
    visibility = ["//tests:__subpackages__"],
    deps = [
        "//compiler_gym/datasets:dataset",
        "//compiler_gym/util",
    ],
)

py_library(
    name = "llvm_env",
    srcs = ["llvm_env.py"],
    data = [
        "//compiler_gym/envs/llvm/service/passes:actions_genfiles",
    ],
    deps = [
        ":benchmarks",
        ":datasets",
        ":llvm_rewards",
        "//compiler_gym/envs:compiler_env",
        "//compiler_gym/spaces",
        "//compiler_gym/third_party/autophase",
        "//compiler_gym/third_party/inst2vec",
    ],
)

py_library(
    name = "llvm_rewards",
    srcs = ["llvm_rewards.py"],
    deps = [
        "//compiler_gym/service",
        "//compiler_gym/spaces",
        "//compiler_gym/views",
    ],
)

genrule(
    name = "specs",
    srcs = ["//compiler_gym/envs/llvm/service:compiler_gym-llvm-service"],
    outs = ["specs.py"],
    cmd = "$(location :make_specs) $(location //compiler_gym/envs/llvm/service:compiler_gym-llvm-service) $@",
    tools = [":make_specs"],
)

py_binary(
    name = "make_specs",
    srcs = ["make_specs.py"],
    deps = [":llvm_env"],
)
