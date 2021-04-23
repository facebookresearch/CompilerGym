# This package exposes the LLVM optimization pipeline as a CompilerGym service.
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

# This target includes the service binary and its runtime dependencies and
# libraries.
filegroup(
    name = "service",
    srcs = [
        ":compiler_gym-llvm-service",
    ] + select({
        "@llvm//:darwin": [],
        "//conditions:default": [
            "//compiler_gym/envs/llvm/service:libLLVMPolly",
        ],
    }),
    visibility = ["//visibility:public"],
)

# Post-build shenanigans to strip the absolute path to LLVMPolly.so for linux
# builds.
genrule(
    name = "compiler_gym-llvm-service-bin",
    srcs = [":compiler_gym-llvm-service-prelinked"],
    outs = ["compiler_gym-llvm-service"],
    cmd = select({
        "@llvm//:darwin": (
            "cp $(location :compiler_gym-llvm-service-prelinked) $@"
        ),
        "//conditions:default": (
            "cp $(location :compiler_gym-llvm-service-prelinked) $@ && " +
            "chmod 666 $@ && " +
            "patchelf --set-rpath '$$ORIGIN' $@ && " +
            "chmod 555 $@"
        ),
    }),
    visibility = ["//compiler_gym/envs/llvm:__subpackages__"],
)

# This target copies the LLVMPolly.so file from LLVM to libLLVMPolly.so.
genrule(
    name = "libLLVMPolly",
    srcs = [
        "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:all_files",
        "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:clang",
    ],
    outs = ["libLLVMPolly.so"],
    cmd = "cp $$(dirname $(location @clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:clang))/../lib/LLVMPolly.so $@",
    visibility = ["//visibility:public"],
)

# Do not use this target directly, use the compiler_gym-llvm_service target
# defined above instead.
cc_binary(
    name = "compiler_gym-llvm-service-prelinked",
    srcs = ["RunService.cc"],
    deps = [
        ":LlvmService",
        "//compiler_gym/util:RunService",
    ],
)

cc_library(
    name = "ActionSpace",
    srcs = [
        "ActionSpace.cc",
    ],
    hdrs = [
        "ActionSpace.h",
        "//compiler_gym/envs/llvm/service/passes:ActionEnum.h",
    ],
    visibility = ["//tests:__subpackages__"],
    deps = [
        "//compiler_gym/service/proto:compiler_gym_service_cc",
        "//compiler_gym/util:EnumUtil",
        "@magic_enum",
    ],
)

cc_library(
    name = "Benchmark",
    srcs = ["Benchmark.cc"],
    hdrs = ["Benchmark.h"],
    visibility = [
        "//compiler_gym/envs/llvm/service:__subpackages__",
        "//tests:__subpackages__",
    ],
    deps = [
        ":Cost",
        "@boost//:filesystem",
        "@com_github_grpc_grpc//:grpc++",
        "@fmt",
        "@glog",
        "@llvm//10.0.0",
    ],
)

cc_library(
    name = "BenchmarkFactory",
    srcs = ["BenchmarkFactory.cc"],
    hdrs = ["BenchmarkFactory.h"],
    visibility = [
        "//compiler_gym/envs/llvm/service:__subpackages__",
        "//tests:__subpackages__",
    ],
    deps = [
        ":Benchmark",
        ":Cost",
        "//compiler_gym/util:GrpcStatusMacros",
        "//compiler_gym/util:RunfilesPath",
        "//compiler_gym/util:StrLenConstexpr",
        "@boost//:filesystem",
        "@com_github_grpc_grpc//:grpc++",
        "@fmt",
        "@glog",
        "@llvm//10.0.0",
    ],
)

cc_library(
    name = "Cost",
    srcs = ["Cost.cc"],
    hdrs = ["Cost.h"],
    deps = [
        "//compiler_gym/util:GrpcStatusMacros",
        "//compiler_gym/util:RunfilesPath",
        "//compiler_gym/util:Unreachable",
        "@boost//:filesystem",
        "@com_github_grpc_grpc//:grpc++",
        "@fmt",
        "@glog",
        "@llvm//10.0.0",
        "@magic_enum",
        "@subprocess",
    ],
)

cc_library(
    name = "LlvmService",
    srcs = ["LlvmService.cc"],
    hdrs = ["LlvmService.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":Benchmark",
        ":BenchmarkFactory",
        ":Cost",
        ":LlvmSession",
        ":ObservationSpaces",
        "//compiler_gym/service/proto:compiler_gym_service_cc",
        "//compiler_gym/util:GrpcStatusMacros",
        "//compiler_gym/util:Version",
        "@boost//:filesystem",
        "@llvm//10.0.0",
    ],
)

cc_library(
    name = "LlvmSession",
    srcs = [
        "LlvmSession.cc",
    ],
    hdrs = [
        "LlvmSession.h",
        "//compiler_gym/envs/llvm/service/passes:ActionHeaders.h",
        "//compiler_gym/envs/llvm/service/passes:ActionSwitch.h",
    ],
    copts = [
        "-DGOOGLE_PROTOBUF_NO_RTTI",
        "-fno-rtti",
    ],
    visibility = ["//tests:__subpackages__"],
    deps = [
        ":ActionSpace",
        ":Benchmark",
        ":Cost",
        ":ObservationSpaces",
        "//compiler_gym/service/proto:compiler_gym_service_cc_grpc",
        "//compiler_gym/third_party/autophase:InstCount",
        "//compiler_gym/third_party/cpuinfo",
        "//compiler_gym/util:EnumUtil",
        "//compiler_gym/util:GrpcStatusMacros",
        "//compiler_gym/util:RunfilesPath",
        "@boost//:filesystem",
        "@fmt",
        "@glog",
        "@llvm//10.0.0",
        "@magic_enum",
        "@nlohmann_json//:json",
        "@programl//programl/graph/format:node_link_graph",
        "@programl//programl/ir/llvm",
        "@programl//programl/proto:programl_cc",
        "@subprocess",
    ],
)

cc_library(
    name = "ObservationSpaces",
    srcs = ["ObservationSpaces.cc"],
    hdrs = ["ObservationSpaces.h"],
    visibility = ["//tests:__subpackages__"],
    deps = [
        "//compiler_gym/service/proto:compiler_gym_service_cc",
        "//compiler_gym/third_party/llvm:InstCount",
        "//compiler_gym/util:EnumUtil",
        "@glog",
        "@magic_enum",
        "@nlohmann_json//:json",
        "@programl//programl/graph/format:node_link_graph",
        "@programl//programl/proto:programl_cc",
    ],
)
