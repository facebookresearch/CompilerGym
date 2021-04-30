# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_binary", "py_test")

py_binary(
    name = "action_sensitivity_analysis",
    srcs = ["action_sensitivity_analysis.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":sensitivity_analysis_eval",
        "//compiler_gym/envs",
        "//compiler_gym/util",
        "//compiler_gym/util/flags:benchmark_from_flags",
        "//compiler_gym/util/flags:env_from_flags",
    ],
)

py_test(
    name = "action_sensitivity_analysis_test",
    srcs = ["action_sensitivity_analysis_test.py"],
    deps = [
        ":action_sensitivity_analysis",
        ":sensitivity_analysis_eval",
        "//tests:test_main",
        "//tests/pytest_plugins:common",
    ],
)

py_binary(
    name = "benchmark_sensitivity_analysis",
    srcs = ["benchmark_sensitivity_analysis.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":sensitivity_analysis_eval",
        "//compiler_gym/envs",
        "//compiler_gym/service/proto",
        "//compiler_gym/util",
        "//compiler_gym/util/flags:benchmark_from_flags",
        "//compiler_gym/util/flags:env_from_flags",
    ],
)

py_test(
    name = "benchmark_sensitivity_analysis_test",
    srcs = ["benchmark_sensitivity_analysis_test.py"],
    deps = [
        ":benchmark_sensitivity_analysis",
        ":sensitivity_analysis_eval",
        "//tests:test_main",
        "//tests/pytest_plugins:common",
    ],
)

py_binary(
    name = "sensitivity_analysis_eval",
    srcs = ["sensitivity_analysis_eval.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym/util",
        "//compiler_gym/util/flags:output_dir",
    ],
)
