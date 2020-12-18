# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Extensions to OpenAI gym's spaces.

py_library(
    name = "spaces",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":commandline",
        ":named_discrete",
        ":scalar",
        ":sequence",
    ],
)

py_library(
    name = "commandline",
    srcs = ["commandline.py"],
    deps = [
        ":named_discrete",
    ],
)

py_library(
    name = "named_discrete",
    srcs = ["named_discrete.py"],
)

py_library(
    name = "scalar",
    srcs = ["scalar.py"],
)

py_library(
    name = "sequence",
    srcs = ["sequence.py"],
)
