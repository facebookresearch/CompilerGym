# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

py_binary(
    name = "loops-opt-service-py",
    srcs = ["loops_opt_service.py"],
    data = [
        "//compiler_gym/third_party/autophase:compute_autophase-prelinked",
        "//compiler_gym/third_party/programl:compute_programl",
        "//examples/loop_optimizations_service/opt_loops",
    ],
    main = "loops_opt_service.py",
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym/service",
        "//compiler_gym/service/proto",
        "//compiler_gym/service/runtime",
        "//compiler_gym/spaces",
        "//compiler_gym/third_party/autophase",
        "//compiler_gym/third_party/inst2vec",
        "//compiler_gym/third_party/llvm",
    ],
)
