# This package contains scripts for extracting passes from the LLVM source tree
# and converting them to an action space for reinforcement learning.
#
# These scripts are used to programatically generate C++ headers and sources
# that are then used to compile the LLVM compiler service.
load("@rules_cc//cc:defs.bzl", "cc_library")

filegroup(
    name = "files",
    srcs = [
        "flag_descriptions.txt",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "headers",
    hdrs = [
        "ActionEnum.h",
        "ActionHeaders.h",
        "ActionSwitch.h",
    ],
    visibility = ["//visibility:public"],
)
