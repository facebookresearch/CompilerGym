# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "py_converters_test",
    timeout = "short",
    srcs = ["py_converters_test.py"],
    deps = [
        "//compiler_gym/service/proto",
        "//compiler_gym/spaces",
        "//tests:test_main",
    ],
)

cc_test(
    name = "proto_cc_test",
    srcs = ["ProtoTest.cc"],
    deps = [
        "//compiler_gym/service/proto:proto_cc",
        "//tests:TestMain",
        "@gtest",
    ],
)
