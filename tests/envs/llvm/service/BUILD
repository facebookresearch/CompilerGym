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

cc_test(
    name = "RewardSpacesTest",
    srcs = ["RewardSpacesTest.cc"],
    deps = [
        "//compiler_gym/envs/llvm/service:RewardSpaces",
        "//tests:TestMacros",
        "//tests:TestMain",
        "@magic_enum",
    ],
)
