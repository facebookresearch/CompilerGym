py_library(
    name = "py_core",
    srcs = [
        "__init__.py",
        "core.py",
    ],
    visibility = ["//visiblity:public"],
    deps = [
        "//compiler_gym/service/proto",
    ],
)

cc_library(
    name = "cc_core",
    srcs = ["Core.cc"],
    hdrs = ["Core.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cc_core_impl",
    ],
)

cc_library(
    name = "cc_core_impl",
    srcs = [
        "BenchmarkProtoCache.cc",
        "CreateAndRunCompilerGymService.cc",
    ],
    hdrs = [
        "BenchmarkProtoCache.h",
        "CompilerGymServicer.h",
        "CreateAndRunCompilerGymService.h",
    ],
    deps = [
        "//compiler_gym/service/proto:compiler_gym_service_cc",
        "//compiler_gym/service/proto:compiler_gym_service_cc_grpc",
        "//compiler_gym/util:GrpcStatusMacros",
        "//compiler_gym/util:Unreachable",
        "//compiler_gym/util:Version",
        "@boost//:filesystem",
        "@com_github_grpc_grpc//:grpc++",
        "@fmt",
        "@gflags",
        "@glog",
    ],
)
