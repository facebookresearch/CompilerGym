# This package contains fuzzers - tests that generate random inputs.
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "llvm_cbench_validate_fuzz_test",
    srcs = ["llvm_cbench_validate_fuzz_test.py"],
    tags = ["manual"],
    deps = [
        "//compiler_gym",
        "//tests:test_main",
        "//tests/pytest_plugins:llvm",
    ],
)

py_test(
    name = "llvm_commandline_opt_equivalence_fuzz_test",
    srcs = ["llvm_commandline_opt_equivalence_fuzz_test.py"],
    tags = ["manual"],
    deps = [
        "//compiler_gym",
        "//tests:test_main",
        "//tests/pytest_plugins:common",
        "//tests/pytest_plugins:llvm",
        "//tests/pytest_plugins:random_util",
    ],
)

py_test(
    name = "llvm_deterministic_action_fuzz_test",
    srcs = ["llvm_deterministic_action_fuzz_test.py"],
    tags = ["manual"],
    deps = [
        "//compiler_gym",
        "//tests:test_main",
        "//tests/pytest_plugins:llvm",
    ],
)

py_test(
    name = "llvm_fork_env_fuzz_test",
    srcs = ["llvm_fork_env_fuzz_test.py"],
    tags = ["manual"],
    deps = [
        "//compiler_gym",
        "//tests:test_main",
        "//tests/pytest_plugins:llvm",
    ],
)

py_test(
    name = "benchmarks_random_actions_test",
    timeout = "long",
    srcs = ["benchmarks_random_actions_test.py"],
    shard_count = 4,
    tags = ["manual"],
    deps = [
        "//compiler_gym",
        "//compiler_gym/envs",
        "//compiler_gym/third_party/autophase",
        "//tests:test_main",
        "//tests/pytest_plugins:llvm",
    ],
)

py_test(
    name = "llvm_trajectory_replay_fuzz_test",
    srcs = ["llvm_trajectory_replay_fuzz_test.py"],
    tags = ["manual"],
    deps = [
        "//compiler_gym",
        "//tests:test_main",
        "//tests/pytest_plugins:llvm",
        "//tests/pytest_plugins:random_util",
    ],
)

py_test(
    name = "llvm_stress_fuzz_test",
    timeout = "long",
    srcs = ["llvm_stress_fuzz_test.py"],
    shard_count = 20,
    tags = ["manual"],
    deps = [
        "//compiler_gym",
        "//tests:test_main",
        "//tests/pytest_plugins:llvm",
        "//tests/pytest_plugins:random_util",
    ],
)
