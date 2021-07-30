# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_foreign_cc//tools/build_defs:configure.bzl", "configure_make")

# Funnel the Csmith sources through a no-op filegroup to fix an issue in
# collecting the generated files for use in `data` attributes of some targets.
# See: https://github.com/bazelbuild/rules_foreign_cc/issues/619
filegroup(
    name = "all",
    srcs = [":csmith"],
    visibility = ["//visibility:public"],
)

configure_make(
    name = "csmith",
    binaries = ["csmith"],
    lib_source = "@csmith//:all",
)
