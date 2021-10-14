# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

# Here we bundle up the new compiler service and data dependencies as a python
# library. To use these example services you can depend on this target. See
# below for examples.
py_library(
    name = "example_compiler_gym_service",
    srcs = ["__init__.py"],
    data = [
        "//examples/example_compiler_gym_service/service_cc:compiler_gym-example-service-cc",
        "//examples/example_compiler_gym_service/service_py:compiler_gym-example-service-py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym/datasets",
        "//compiler_gym/spaces",
        "//compiler_gym/util",
    ],
)

# This is a scrip that demonstrates how to use the example services. To run it,
# use:  $ bazel run -c opt //examples/example_compiler_gym_service:demo
py_binary(
    name = "demo",
    srcs = ["demo.py"],
    deps = [
        ":example_compiler_gym_service",
        "//compiler_gym",
    ],
)

# This is a set of unit tests for the example services. To run the tests, use:
# $ bazel test //examples/example_compiler_gym_service:env_tests
py_test(
    name = "env_tests",
    srcs = ["env_tests.py"],
    deps = [
        ":example_compiler_gym_service",
        "//compiler_gym",
        "//tests:test_main",
    ],
)
