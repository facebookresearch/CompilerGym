# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

genrule(
    name = "make_macos_bits_stdcxx",
    srcs = ["macos_bits_stdc++.h"],
    outs = ["macos_bits_stdcxx.py"],
    cmd = (
        "echo 'MACOS_BITS_STDCXX = \"\"\"' > $@ &&" +
        "cat $(location :macos_bits_stdc++.h) >> $@ &&" +
        "echo '\"\"\"' >> $@"
    ),
    visibility = ["//visibility:public"],
)
