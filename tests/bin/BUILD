# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "datasets_bin_test",
    srcs = ["datasets_bin_test.py"],
    deps = [
        "//compiler_gym/bin:datasets",
        "//tests:test_main",
        "//tests/pytest_plugins:common",
    ],
)

py_test(
    name = "manual_env_bin_test",
    timeout = "short",
    srcs = ["manual_env_bin_test.py"],
    flaky = 1,
    deps = [
        "//compiler_gym/bin:manual_env",
        "//compiler_gym/util",
        "//tests:test_main",
    ],
)

py_test(
    name = "service_bin_test",
    srcs = ["service_bin_test.py"],
    deps = [
        "//compiler_gym",
        "//compiler_gym/bin:service",
        "//tests:test_main",
    ],
)

py_test(
    name = "validate_bin_test",
    timeout = "long",
    srcs = ["validate_bin_test.py"],
    shard_count = 4,
    deps = [
        "//compiler_gym",
        "//compiler_gym/bin:validate",
        "//tests:test_main",
        "//tests/pytest_plugins:common",
    ],
)
