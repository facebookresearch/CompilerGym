# This package exposes the LLVM optimization pipeline as a CompilerGym service.

cc_binary(
    name = "service",
    srcs = ["RunService.cc"],
    visibility = ["//visibility:public"],
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
    visibility = ["//compiler_gym/envs/llvm/service:__subpackages__"],
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
    visibility = ["//compiler_gym/envs/llvm/service:__subpackages__"],
    deps = [
        ":Benchmark",
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
    data = [
        "//compiler_gym/third_party/llvm:clang",
    ],
    deps = [
        ":RewardSpaces",
        "//compiler_gym/util:GrpcStatusMacros",
        "//compiler_gym/util:RunfilesPath",
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
    name = "LlvmEnvironment",
    srcs = [
        "LlvmEnvironment.cc",
    ],
    hdrs = [
        "LlvmEnvironment.h",
        "//compiler_gym/envs/llvm/service/passes:ActionHeaders.h",
        "//compiler_gym/envs/llvm/service/passes:ActionSwitch.h",
    ],
    copts = [
        "-DGOOGLE_PROTOBUF_NO_RTTI",
        "-fno-rtti",
    ],
    deps = [
        ":ActionSpace",
        ":Benchmark",
        ":Cost",
        ":ObservationSpaces",
        ":RewardSpaces",
        "//compiler_gym/service/proto:compiler_gym_service_cc_grpc",
        "//compiler_gym/third_party/autophase:InstCount",
        "//compiler_gym/third_party/cpuinfo",
        "//compiler_gym/util:EnumUtil",
        "//compiler_gym/util:GrpcStatusMacros",
        "@boost//:filesystem",
        "@fmt",
        "@glog",
        "@llvm//10.0.0",
        "@magic_enum",
        "@nlohmann_json//:json",
        "@programl//programl/graph/format:node_link_graph",
        "@programl//programl/ir/llvm",
        "@programl//programl/proto:programl_cc",
    ],
)

cc_library(
    name = "LlvmService",
    srcs = ["LlvmService.cc"],
    hdrs = ["LlvmService.h"],
    deps = [
        ":Benchmark",
        ":BenchmarkFactory",
        ":Cost",
        ":LlvmEnvironment",
        ":ObservationSpaces",
        ":RewardSpaces",
        "//compiler_gym/service/proto:compiler_gym_service_cc",
        "//compiler_gym/util:GrpcStatusMacros",
        "@boost//:filesystem",
    ],
)

cc_library(
    name = "ObservationSpaces",
    srcs = ["ObservationSpaces.cc"],
    hdrs = ["ObservationSpaces.h"],
    visibility = ["//tests:__subpackages__"],
    deps = [
        "//compiler_gym/service/proto:compiler_gym_service_cc",
        "//compiler_gym/util:EnumUtil",
        "@magic_enum",
    ],
)

cc_library(
    name = "RewardSpaces",
    srcs = ["RewardSpaces.cc"],
    hdrs = ["RewardSpaces.h"],
    visibility = ["//tests:__subpackages__"],
    deps = [
        "//compiler_gym/service/proto:compiler_gym_service_cc",
        "//compiler_gym/util:EnumUtil",
        "@magic_enum",
    ],
)
