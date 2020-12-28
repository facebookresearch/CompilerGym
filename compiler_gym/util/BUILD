# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

py_library(
    name = "util",
    srcs = [
        "__init__.py",
        "capture_output.py",
        "download.py",
        "logs.py",
        "registration.py",
        "runfiles_path.py",
        "shell_format.py",
        "statistics.py",
        "tabulate.py",
        "timer.py",
        "user_input.py",
        "version.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@rules_python//python/runfiles",
    ],
)

cc_library(
    name = "EnumUtil",
    srcs = ["EnumUtil.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_grpc_grpc//:grpc++",
        "@fmt",
        "@magic_enum",
    ],
)

cc_library(
    name = "GrpcStatusMacros",
    hdrs = ["GrpcStatusMacros.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_grpc_grpc//:grpc++",
    ],
)

cc_library(
    name = "RunfilesPath",
    srcs = ["RunfilesPath.cc"],
    hdrs = ["RunfilesPath.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@boost//:filesystem",
    ],
)

cc_library(
    name = "RunService",
    srcs = ["RunService.cc"],
    hdrs = ["RunService.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":RunServiceImpl",
        "@boost//:filesystem",
        "@gflags",
        "@glog",
    ],
)

cc_library(
    name = "RunServiceImpl",
    hdrs = ["RunServiceImpl.h"],
    deps = [
        "@boost//:filesystem",
        "@com_github_grpc_grpc//:grpc++",
        "@glog",
    ],
)

cc_library(
    name = "StrLenConstexpr",
    hdrs = ["StrLenConstexpr.h"],
    visibility = ["//visibility:public"],
)

genrule(
    name = "make_version",
    srcs = ["//:VERSION"],
    outs = ["version.py"],
    cmd = "echo \"__version__ = \\\"$$(cat $<)\\\"\" > $@",
)

cc_library(
    name = "Version",
    hdrs = ["Version.h"],
    visibility = ["//visibility:public"],
)

genrule(
    name = "make_version_header",
    srcs = ["//:VERSION"],
    outs = ["Version.h"],
    cmd = "echo \"#define COMPILER_GYM_VERSION \\\"$$(cat $<)\\\"\" > $@",
)
