# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "flags",
    srcs = [
        "benchmark_from_flags.py",
        "env_from_flags.py",
        "episode_length.py",
        "episodes.py",
        "learning_rate.py",
        "nproc.py",
        "output_dir.py",
        "seed.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym/datasets",
        "//compiler_gym/envs",
        "//compiler_gym/service",
        "//compiler_gym/service/proto",
    ],
)
