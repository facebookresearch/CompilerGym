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
    configure_env_vars = {
        # Workaround error with libtool usage on macOS. See:
        # https://github.com/bazelbuild/rules_foreign_cc/issues/185
        "AR": "/usr/bin/ar",
        # Csmith uses decreated stdlib functions like std::bind2nd().
        "CXXFLAGS": "-D_LIBCPP_ENABLE_CXX17_REMOVED_FEATURES",
    },
    # Workaround a strange bug where the srand48_deterministic test returns
    # true on macOS, although this only available and needed for OpenBSD.
    configure_options = ["ac_cv_func_srand48_deterministic=no"],
    lib_source = "@csmith//:all",
)
