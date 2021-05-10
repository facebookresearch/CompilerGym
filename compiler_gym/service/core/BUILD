py_library(
    name = "py_core",
    srcs = [
        "__init__.py",
        "compiler_gym_servicer.py",
        "core.py",
        "proto2py.py",
        "run.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym/service/proto",
    ],
)

cc_library(
    name = "cc_core",
    srcs = [
        "BenchmarkProtoCache.cc",
        "Core.cc",
        "CreateAndRunCompilerGymService.cc",
    ],
    hdrs = [
        "BenchmarkProtoCache.h",
        "CompilerGymServicer.h",
        "Core.h",
        "CreateAndRunCompilerGymService.h",
        "Run.h",
    ],
    visibility = ["//visibility:public"],
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

py_library(
    name = "proto2py",
    srcs = ["proto2py.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym/service/proto",
    ],
)
