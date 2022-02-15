# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the LICENSE file
# in the root directory of this source tree.
#
# This package exposes the LLVM optimization pipeline as a CompilerGym service.
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

package(default_visibility = [
    "//compiler_gym/envs/llvm:__subpackages__",
    "//tests:__subpackages__",
])

# This target includes the service binary and its runtime dependencies and
# libraries.
filegroup(
    name = "service",
    srcs = [
        ":compiler_gym-llvm-service",
    ] + select({
        "@llvm//:darwin": [],
        "//conditions:default": [
            ":libLLVMPolly",
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
        ":LlvmSession",
        "//compiler_gym/service/runtime:cc_runtime",
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
    deps = [
        "//compiler_gym/service/proto:compiler_gym_service_cc",
        "//compiler_gym/util:EnumUtil",
        "//compiler_gym/util:Unreachable",
        "@fmt",
        "@magic_enum",
    ],
)

cc_library(
    name = "Benchmark",
    srcs = ["Benchmark.cc"],
    hdrs = ["Benchmark.h"],
    deps = [
        ":BenchmarkDynamicConfig",
        ":Cost",
        "//compiler_gym/service/proto:compiler_gym_service_cc",
        "//compiler_gym/util:GrpcStatusMacros",
        "//compiler_gym/util:RunfilesPath",
        "//compiler_gym/util:Subprocess",
        "@boost//:filesystem",
        "@com_github_grpc_grpc//:grpc++",
        "@fmt",
        "@glog",
        "@llvm//10.0.0",
    ],
)

cc_library(
    name = "BenchmarkDynamicConfig",
    srcs = ["BenchmarkDynamicConfig.cc"],
    hdrs = ["BenchmarkDynamicConfig.h"],
    deps = [
        "//compiler_gym/service/proto:compiler_gym_service_cc",
        "//compiler_gym/util:RunfilesPath",
        "//compiler_gym/util:Subprocess",
        "@boost//:filesystem",
    ],
)

cc_library(
    name = "BenchmarkFactory",
    srcs = ["BenchmarkFactory.cc"],
    hdrs = ["BenchmarkFactory.h"],
    deps = [
        ":Benchmark",
        ":BenchmarkDynamicConfig",
        ":Cost",
        "//compiler_gym/service/proto:compiler_gym_service_cc",
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

filegroup(
    name = "compute_observation-files",
    srcs = [
        ":compute_observation",
    ] + select({
        "@llvm//:darwin": [],
        "//conditions:default": [
            ":libLLVMPolly",
        ],
    }),
)

cc_binary(
    name = "compute_observation-prelinked",
    srcs = ["ComputeObservation.cc"],
    copts = [
        "-DGOOGLE_PROTOBUF_NO_RTTI",
        "-fno-rtti",
    ],
    deps = [
        ":BenchmarkFactory",
        ":Observation",
        ":ObservationSpaces",
        "//compiler_gym/service/proto:compiler_gym_service_cc",
        "@boost//:filesystem",
        "@glog",
        "@llvm//10.0.0",
        "@magic_enum",
    ],
)

genrule(
    name = "compute_observation-bin",
    srcs = [":compute_observation-prelinked"],
    outs = ["compute_observation"],
    cmd = select({
        "@llvm//:darwin": (
            "cp $(location :compute_observation-prelinked) $@"
        ),
        "//conditions:default": (
            "cp $(location :compute_observation-prelinked) $@ && " +
            "chmod 666 $@ && " +
            "patchelf --set-rpath '$$ORIGIN' $@ && " +
            "chmod 555 $@"
        ),
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "Cost",
    srcs = ["Cost.cc"],
    hdrs = ["Cost.h"],
    deps = [
        ":BenchmarkDynamicConfig",
        "//compiler_gym/util:GrpcStatusMacros",
        "//compiler_gym/util:RunfilesPath",
        "//compiler_gym/util:Subprocess",
        "//compiler_gym/util:Unreachable",
        "@boost//:filesystem",
        "@boost//:process",
        "@com_github_grpc_grpc//:grpc++",
        "@fmt",
        "@glog",
        "@llvm//10.0.0",
        "@magic_enum",
    ],
)

cc_library(
    name = "LlvmSession",
    srcs = ["LlvmSession.cc"],
    hdrs = [
        "LlvmSession.h",
        "//compiler_gym/envs/llvm/service/passes:ActionHeaders.h",
        "//compiler_gym/envs/llvm/service/passes:ActionSwitch.h",
    ],
    copts = [
        "-DGOOGLE_PROTOBUF_NO_RTTI",
        "-fno-rtti",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":ActionSpace",
        ":Benchmark",
        ":BenchmarkFactory",
        ":Cost",
        ":Observation",
        ":ObservationSpaces",
        "//compiler_gym/service:CompilationSession",
        "//compiler_gym/service/proto:compiler_gym_service_cc_grpc",
        "//compiler_gym/third_party/autophase:InstCount",
        "//compiler_gym/third_party/cpuinfo",
        "//compiler_gym/util:EnumUtil",
        "//compiler_gym/util:GrpcStatusMacros",
        "//compiler_gym/util:RunfilesPath",
        "@boost//:filesystem",
        "@boost//:process",
        "@fmt",
        "@glog",
        "@llvm//10.0.0",
        "@magic_enum",
        "@nlohmann_json//:json",
        "@programl//programl/graph/format:node_link_graph",
        "@programl//programl/ir/llvm:llvm-10",
        "@programl//programl/proto:programl_cc",
    ],
)

cc_library(
    name = "Observation",
    srcs = ["Observation.cc"],
    hdrs = ["Observation.h"],
    deps = [
        ":Benchmark",
        ":Cost",
        ":ObservationSpaces",
        "//compiler_gym/service/proto:compiler_gym_service_cc_grpc",
        "//compiler_gym/third_party/autophase:InstCount",
        "//compiler_gym/third_party/cpuinfo",
        "//compiler_gym/util:GrpcStatusMacros",
        "@boost//:filesystem",
        "@glog",
        "@llvm//10.0.0",
        "@magic_enum",
        "@nlohmann_json//:json",
        "@programl//programl/graph/format:node_link_graph",
        "@programl//programl/ir/llvm:llvm-10",
        "@programl//programl/proto:programl_cc",
    ],
)

cc_library(
    name = "ObservationSpaces",
    srcs = ["ObservationSpaces.cc"],
    hdrs = ["ObservationSpaces.h"],
    deps = [
        ":Benchmark",
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

# The strip-optnone-attribute binary is a utility for stripping the "optnone"
# function attribute from LLVM bitcode files. It is used for preparing datasets
# of bitcodes.

filegroup(
    name = "strip-optnone-attribute-files",
    srcs = [
        ":strip-optnone-attribute",
    ] + select({
        "@llvm//:darwin": [],
        "//conditions:default": [
            ":libLLVMPolly",
        ],
    }),
)

cc_binary(
    name = "strip-optnone-attribute-prelinked",
    srcs = ["StripOptNoneAttribute.cc"],
    copts = [
        "-DGOOGLE_PROTOBUF_NO_RTTI",
        "-fno-rtti",
    ],
    deps = [
        ":BenchmarkFactory",
        "//compiler_gym/service/proto:compiler_gym_service_cc",
        "//compiler_gym/util:GrpcStatusMacros",
        "@boost//:filesystem",
        "@glog",
        "@llvm//10.0.0",
        "@magic_enum",
    ],
)

genrule(
    name = "strip-optnone-bin",
    srcs = [":strip-optnone-attribute-prelinked"],
    outs = ["strip-optnone-attribute"],
    cmd = select({
        "@llvm//:darwin": (
            "cp $(location :strip-optnone-attribute-prelinked) $@"
        ),
        "//conditions:default": (
            "cp $(location :strip-optnone-attribute-prelinked) $@ && " +
            "chmod 666 $@ && " +
            "patchelf --set-rpath '$$ORIGIN' $@ && " +
            "chmod 555 $@"
        ),
    }),
    visibility = ["//visibility:public"],
)
