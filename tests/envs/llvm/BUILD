# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

py_test(
    name = "all_actions_single_step_test",
    timeout = "long",
    srcs = ["all_actions_single_step_test.py"],
    shard_count = 8,
    deps = [
        ":fixtures",
        "//compiler_gym/envs",
        "//tests:test_main",
    ],
)

py_test(
    name = "all_benchmarks_init_close_test",
    timeout = "long",
    srcs = ["all_benchmarks_init_close_test.py"],
    shard_count = 4,
    deps = [
        ":fixtures",
        "//compiler_gym/envs",
        "//tests:test_main",
    ],
)

py_test(
    name = "autophase_test",
    timeout = "short",
    srcs = ["autophase_test.py"],
    deps = [
        ":fixtures",
        "//compiler_gym/envs",
        "//tests:test_main",
    ],
)

py_test(
    name = "benchmarks_random_actions_test",
    timeout = "eternal",
    srcs = ["benchmarks_random_actions_test.py"],
    shard_count = 4,
    deps = [
        ":fixtures",
        "//compiler_gym",
        "//compiler_gym/envs",
        "//tests:test_main",
    ],
)

py_test(
    name = "custom_benchmarks_test",
    srcs = ["custom_benchmarks_test.py"],
    data = [
        "//compiler_gym/third_party/cBench:crc32",
    ],
    deps = [
        ":fixtures",
        "//compiler_gym/envs",
        "//compiler_gym/service/proto",
        "//compiler_gym/util",
        "//tests:test_main",
    ],
)

py_library(
    name = "fixtures",
    testonly = 1,
    srcs = ["fixtures.py"],
    data = [
        "//compiler_gym/envs/llvm/service",
        "//compiler_gym/envs/llvm/service/passes:actions_genfiles",
        "//compiler_gym/third_party/cBench:benchmarks_list",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym",
        "//compiler_gym/envs",
        "//compiler_gym/util",
        "//tests:test_main",
    ],
)

py_test(
    name = "fresh_environment_observation_reward_test",
    srcs = ["fresh_environment_observation_reward_test.py"],
    shard_count = 4,
    deps = [
        ":fixtures",
        "//compiler_gym/envs",
        "//tests:test_main",
    ],
)

py_test(
    name = "llvm_benchmarks_test",
    srcs = ["llvm_benchmarks_test.py"],
    deps = [
        ":fixtures",
        "//compiler_gym/envs",
        "//compiler_gym/service/proto",
        "//tests:test_main",
    ],
)

py_test(
    name = "llvm_env_test",
    srcs = ["llvm_env_test.py"],
    deps = [
        ":fixtures",
        "//compiler_gym",
        "//tests:test_main",
    ],
)

py_test(
    name = "multiprocessing_test",
    timeout = "short",
    srcs = ["multiprocessing_test.py"],
    deps = [
        ":fixtures",
        "//compiler_gym",
        "//tests:test_main",
    ],
)

py_test(
    name = "observation_spaces_test",
    srcs = ["observation_spaces_test.py"],
    deps = [
        ":fixtures",
        "//compiler_gym/envs",
        "//tests:test_main",
    ],
)

py_test(
    name = "reward_spaces_test",
    srcs = ["reward_spaces_test.py"],
    deps = [
        ":fixtures",
        "//compiler_gym/envs",
        "//tests:test_main",
    ],
)

py_test(
    name = "threading_test",
    timeout = "short",
    srcs = ["threading_test.py"],
    deps = [
        "//compiler_gym",
        "//tests:test_main",
    ],
)
