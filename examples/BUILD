# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_binary", "py_test")
load("@rules_cc//cc:defs.bzl", "cc_binary")

py_binary(
    name = "actor_critic",
    srcs = ["actor_critic.py"],
    deps = [
        "//compiler_gym",
    ],
)

py_test(
    name = "actor_critic_test",
    timeout = "short",
    srcs = ["actor_critic_test.py"],
    deps = [
        ":actor_critic",
        "//compiler_gym/util",
        "//tests:test_main",
        "//tests/pytest_plugins:common",
    ],
)

py_binary(
    name = "brute_force",
    srcs = ["brute_force.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym/envs",
        "//compiler_gym/service/proto",
        "//compiler_gym/util",
        "//compiler_gym/util/flags:benchmark_from_flags",
        "//compiler_gym/util/flags:env_from_flags",
        "//compiler_gym/util/flags:output_dir",
    ],
)

py_test(
    name = "brute_force_test",
    srcs = ["brute_force_test.py"],
    deps = [
        ":brute_force",
        "//compiler_gym",
        "//tests:test_main",
    ],
)

py_binary(
    name = "explore",
    srcs = ["explore.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym/envs",
        "//compiler_gym/service/proto",
        "//compiler_gym/util",
        "//compiler_gym/util/flags:benchmark_from_flags",
        "//compiler_gym/util/flags:env_from_flags",
    ],
)

py_test(
    name = "explore_test",
    srcs = ["explore_test.py"],
    deps = [
        ":explore",
        "//compiler_gym",
        "//tests:test_main",
    ],
)

py_binary(
    name = "tabular_q",
    srcs = ["tabular_q.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym",
        "//compiler_gym/util",
        "//compiler_gym/util/flags:benchmark_from_flags",
    ],
)

py_test(
    name = "tabular_q_test",
    timeout = "short",
    srcs = ["tabular_q_test.py"],
    deps = [
        ":tabular_q",
        "//compiler_gym/util",
        "//tests:test_main",
        "//tests/pytest_plugins:common",
    ],
)

py_binary(
    name = "random_walk",
    srcs = ["random_walk.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym/envs",
        "//compiler_gym/util",
        "//compiler_gym/util/flags:benchmark_from_flags",
        "//compiler_gym/util/flags:env_from_flags",
    ],
)

py_test(
    name = "random_walk_test",
    srcs = ["random_walk_test.py"],
    deps = [
        ":random_walk",
        "//compiler_gym/util",
        "//tests:test_main",
    ],
)

cc_binary(
    name = "RandomSearch",
    srcs = ["RandomSearch.cc"],
    deps = [
        "//compiler_gym/envs/llvm/service:LlvmService",
        "@boost//:filesystem",
        "@gflags",
        "@glog",
        "@magic_enum",
    ],
)
