# This package contains scripts for extracting passes from the LLVM source tree
# and converting them to an action space for reinforcement learning.
load("@rules_python//python:defs.bzl", "py_binary", "py_library")

genrule(
    name = "passes_list",
    srcs = [
        "@llvm//10.0.0:llvm_src",
        "@llvm//10.0.0:llvm_readme",
    ],
    outs = ["passes_list.csv"],
    cmd = (
        "$(location :extract_passes_from_llvm_source_tree) " +
        "$$(dirname $(location @llvm//10.0.0:llvm_readme))" +
        "> $@"
    ),
    tools = [
        ":extract_passes_from_llvm_source_tree",
    ],
    visibility = ["//visibility:public"],
)

genrule(
    name = "actions_csv",
    srcs = [":passes_list"],
    outs = ["actions.csv"],
    cmd = "$(location :filter_action_space) < $(location :passes_list) > $@",
    tools = [
        ":filter_action_space",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "common",
    srcs = ["common.py"],
)

py_library(
    name = "config",
    srcs = ["config.py"],
    deps = [":common"],
)

py_binary(
    name = "extract_passes_from_llvm_source_tree",
    srcs = ["extract_passes_from_llvm_source_tree.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        ":config",
    ],
)

py_binary(
    name = "make_action_space_genfiles",
    srcs = ["make_action_space_genfiles.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        ":config",
    ],
)

py_binary(
    name = "filter_action_space",
    srcs = ["filter_action_space.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        ":config",
    ],
)

# Generated files defining the LLVM action space.
genrule(
    name = "actions_genfiles",
    srcs = [":actions_csv"],
    outs = [
        "ActionEnum.h",
        "ActionSwitch.h",
        "ActionHeaders.h",
        "actions_list.txt",
        "actions_flags.txt",
        "actions_descriptions.txt",
    ],
    cmd = "$(location :make_action_space_genfiles) $(@D) < $(location :actions_csv)",
    tools = [":make_action_space_genfiles"],
    visibility = ["//visibility:public"],
)
