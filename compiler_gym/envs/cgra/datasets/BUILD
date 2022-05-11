load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "datasets",
    srcs = [
        "__init__.py",
        "dfg_bench.py"
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym/datasets",
        "//compiler_gym/service/proto",
        "//compiler_gym/util",
    ]
)