# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cc_test(
    name = "ActionSpaceTest",
    srcs = ["ActionSpaceTest.cc"],
    deps = [
        "//compiler_gym/envs/llvm/service:ActionSpace",
        "//tests:TestMacros",
        "//tests:TestMain",
        "@gtest",
        "@magic_enum",
    ],
)

# NOTE(https://github.com/facebookresearch/CompilerGym/issues/46): The -gvn-sink
# pass is temporarily disabled.
#
# cc_test(
#     name = "GvnSinkTest",
#     srcs = ["GvnSinkTest.cc"],
#     data = ["//compiler_gym/third_party/cBench:blowfish"],
#     deps = [
#         "//compiler_gym/envs/llvm/service:BenchmarkFactory",
#         "//compiler_gym/envs/llvm/service:LlvmEnvironment",
#         "//compiler_gym/util:GrpcStatusMacros",
#         "//compiler_gym/util:RunfilesPath",
#         "//tests:TestMain",
#         "@boost//:filesystem",
#         "@glog",
#         "@gtest",
#     ],
# )

cc_test(
    name = "ObservationSpacesTest",
    srcs = ["ObservationSpacesTest.cc"],
    deps = [
        "//compiler_gym/envs/llvm/service:ObservationSpaces",
        "//tests:TestMain",
        "@gtest",
        "@magic_enum",
    ],
)
