# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

py_test(
    name = "datasets_test",
    srcs = ["datasets_test.py"],
    deps = [
        "//compiler_gym/bin:datasets",
        "//tests:test_main",
    ],
)

py_test(
    name = "manual_env_test",
    timeout = "short",
    srcs = ["manual_env_test.py"],
    deps = [
        "//compiler_gym/bin:manual_env",
        "//compiler_gym/util",
        "//tests:test_main",
    ],
)

py_test(
    name = "service_test",
    srcs = ["service_test.py"],
    deps = [
        "//compiler_gym",
        "//compiler_gym/bin:service",
        "//tests:test_main",
    ],
)

py_test(
    name = "validate_test",
    timeout = "long",
    srcs = ["validate_test.py"],
    deps = [
        "//compiler_gym",
        "//compiler_gym/bin:validate",
        "//tests:fixtures",
        "//tests:test_main",
    ],
)
