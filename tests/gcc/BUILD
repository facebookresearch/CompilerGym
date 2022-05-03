# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "gcc_bin_test",
    timeout = "short",
    srcs = ["gcc_bin_test.py"],
    deps = [
        "//compiler_gym/envs/gcc",
        "//compiler_gym/errors",
        "//tests:test_main",
        "//tests/pytest_plugins:gcc",
    ],
)

py_test(
    name = "gcc_docker_test",
    timeout = "long",
    srcs = ["gcc_docker_test.py"],
    deps = [
        "//compiler_gym/envs/gcc",
        "//compiler_gym/errors",
        "//tests:test_main",
        "//tests/pytest_plugins:common",
    ],
)

py_test(
    name = "gcc_env_test",
    timeout = "long",
    srcs = ["gcc_env_test.py"],
    shard_count = 4,
    deps = [
        "//compiler_gym/envs/gcc",
        "//compiler_gym/errors",
        "//compiler_gym/spaces",
        "//tests:test_main",
        "//tests/pytest_plugins:common",
        "//tests/pytest_plugins:gcc",
    ],
)
