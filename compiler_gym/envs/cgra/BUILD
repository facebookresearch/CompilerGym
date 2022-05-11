# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_library", "py_test")

py_library(
    name = "cgra",
    srcs = [
        "__init__.py",
        "cgra_rewards.py",
        "Operations.py",
        "DFG.py"
    ],
    data = [
        "//compiler_gym/envs/cgra/service",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym/envs/cgra/datasets",
        "//compiler_gym/errors",
        "//compiler_gym/service:client_service_compiler_env",
        "//compiler_gym/service/runtime",  # Implicit dependency of service.
        "//compiler_gym/util"
    ],
)