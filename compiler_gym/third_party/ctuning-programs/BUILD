# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

genrule(
    name = "mibench_tar",
    srcs = [":mibench"],
    outs = ["mibench-llvm-10.0.0.tar.bz2"],
    cmd = "tar cjfh $@ -C $(@D) mibench",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "mibench",
    srcs = [
        ":mibench-bitcount-1",
        ":mibench-bitcount-2",
        ":mibench-dijkstra",
        ":mibench-fft-1",
        ":mibench-fft-2",
        ":mibench-ghostscript",
        ":mibench-gsm-1",
        ":mibench-gsm-2",
        ":mibench-jpeg-c",
        ":mibench-jpeg-c-1",
        ":mibench-jpeg-c-2",
        ":mibench-jpeg-c-3",
        ":mibench-lame-newmdct-1",
        ":mibench-lame-newmdct-2",
        ":mibench-lame-psymodel",
        ":mibench-lame-quantize-1",
        ":mibench-lame-quantize-2",
        ":mibench-lame-takehiro-1",
        ":mibench-lame-takehiro-2",
        ":mibench-mad-1",
        ":mibench-mad-2",
        ":mibench-pgp-d",
        ":mibench-pgp-e-1",
        ":mibench-pgp-e-2",
        ":mibench-pgp-e-3",
        ":mibench-qsort1",
        ":mibench-rsynth-1",
        ":mibench-rsynth-2",
        ":mibench-susan-e-1",
        ":mibench-susan-e-2",
        ":mibench-susan-s",
        ":mibench-tiff2rgba",
        ":mibench-tiffdither-1",
        ":mibench-tiffdither-2",
        ":mibench-tiffdither-3",
        ":mibench-tiffmedian-1",
        ":mibench-tiffmedian-2",
        ":mibench-tiffmedian-3",
        ":mibench-tiffmedian-4",
        ":mibench-tiffmedian-5",
    ],
)

genrule(
    name = "polybench_tar",
    srcs = [":polybench"],
    outs = ["polybench-llvm-10.0.0.tar.bz2"],
    cmd = "tar cjfh $@ -C $(@D) polybench",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "polybench",
    srcs = [
        ":polybench-2mm",
        ":polybench-3mm",
        ":polybench-adi",
        ":polybench-atax",
        ":polybench-bicg",
        ":polybench-cholesky",
        ":polybench-doitgen",
        ":polybench-durbin",
        ":polybench-dynprog",
        ":polybench-fdtd-1",
        ":polybench-fdtd-2",
        ":polybench-gemm",
        ":polybench-gemver",
        ":polybench-gesummv",
        ":polybench-gramschmidt",
        ":polybench-jacobi-1d",
        ":polybench-jacobi-2d",
        ":polybench-lu",
        ":polybench-medley-floyd",
        ":polybench-medley-reg",
        ":polybench-mvt",
        ":polybench-seidel",
        ":polybench-symm",
        ":polybench-syr2k",
        ":polybench-syrk",
        ":polybench-trisolv",
        ":polybench-trmm",
    ],
)

sh_binary(
    name = "make_bitcode",
    srcs = ["make_bitcode.sh"],
    visibility = ["//visibility:public"],
)

genrule(
    name = "mibench-bitcount-1",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/bitcount-1.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-automotive-bitcount-src-bitcnt-1-codelet-2-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-bitcount-2",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/bitcount-2.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-automotive-bitcount-src-bitcnts-codelet-1-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-qsort1",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/qsort1.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-automotive-qsort1-src-qsort-codelet-1-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-susan-e-1",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/susan-e-1.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-automotive-susan-e-src-susan-codelet-10-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-susan-e-2",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/susan-e-2.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-automotive-susan-e-src-susan-codelet-2-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-susan-s",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/susan-s.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-automotive-susan-s-src-susan-codelet-1-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-jpeg-c-1",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/jpeg-c-1.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-consumer-jpeg-c-src-jcdctmgr-codelet-13-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-jpeg-c-2",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/jpeg-c-2.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-consumer-jpeg-c-src-jchuff-codelet-9-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-jpeg-c-3",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/jpeg-c-3.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-consumer-jpeg-c-src-jfdctint-codelet-2-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-jpeg-c",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/jpeg-c.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-consumer-lame-src-fft-codelet-2-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-lame-newmdct-1",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/lame-newmdct-1.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-consumer-lame-src-newmdct-codelet-10-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-lame-newmdct-2",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/lame-newmdct-2.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-consumer-lame-src-newmdct-codelet-3-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-lame-psymodel",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/lame-psymodel.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-consumer-lame-src-psymodel-codelet-17-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-lame-quantize-1",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/lame-quantize-1.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-consumer-lame-src-quantize-codelet-7-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-lame-quantize-2",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/lame-quantize-2.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-consumer-lame-src-quantize-pvt-codelet-6-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-lame-takehiro-1",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/lame-takehiro-1.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-consumer-lame-src-takehiro-codelet-16-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-lame-takehiro-2",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/lame-takehiro-2.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-consumer-lame-src-takehiro-codelet-5-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-mad-1",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/mad-1.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-consumer-mad-src-layer3-codelet-5-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-mad-2",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/mad-2.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-consumer-mad-src-layer3-codelet-6-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-tiff2rgba",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/tiff2rgba.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-consumer-tiff2rgba-src-tif-predict-codelet-4-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-tiffdither-1",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/tiffdither-1.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-consumer-tiffdither-src-tif-fax3-codelet-8-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-tiffdither-2",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/tiffdither-2.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-consumer-tiffdither-src-tif-fax3-codelet-9-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-tiffdither-3",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/tiffdither-3.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-consumer-tiffdither-src-tiffdither-codelet-1-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-tiffmedian-1",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/tiffmedian-1.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-consumer-tiffmedian-src-tiffmedian-codelet-1-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-tiffmedian-2",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/tiffmedian-2.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-consumer-tiffmedian-src-tiffmedian-codelet-3-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-tiffmedian-3",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/tiffmedian-3.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-consumer-tiffmedian-src-tiffmedian-codelet-4-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-tiffmedian-4",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/tiffmedian-4.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-consumer-tiffmedian-src-tiffmedian-codelet-5-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-tiffmedian-5",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/tiffmedian-5.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-consumer-tiffmedian-src-tiffmedian-codelet-6-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-dijkstra",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/dijkstra.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-network-dijkstra-src-dijkstra-large-codelet-5-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-ghostscript",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/ghostscript.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-office-ghostscript-src-gdevpbm-codelet-1-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-rsynth-1",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/rsynth-1.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-office-rsynth-src-nsynth-codelet-5-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-rsynth-2",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/rsynth-2.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-office-rsynth-src-nsynth-codelet-9-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-pgp-d",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/pgp-d.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-security-pgp-d-src-mpilib-codelet-1-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-pgp-e-1",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/pgp-e-1.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-security-pgp-e-src-mpilib-codelet-1-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-pgp-e-2",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/pgp-e-2.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-security-pgp-e-src-mpilib-codelet-3-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-pgp-e-3",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/pgp-e-3.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-security-pgp-e-src-mpilib-codelet-4-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-fft-1",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/fft-1.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-telecomm-fft-fftmisc-codelet-5-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-fft-2",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/fft-2.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-telecomm-fft-fourierf-codelet-3-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-gsm-1",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/gsm-1.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-telecomm-gsm-src-rpe-codelet-4-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "mibench-gsm-2",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["mibench/gsm-2.bc"],
    cmd = "mkdir -p mibench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/milepost-codelet-mibench-telecomm-gsm-src-short-term-codelet-2-1",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "polybench-2mm",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["polybench/2mm.bc"],
    cmd = "mkdir -p polybench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/polybench-cpu-2mm",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "polybench-3mm",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["polybench/3mm.bc"],
    cmd = "mkdir -p polybench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/polybench-cpu-3mm",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "polybench-adi",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["polybench/adi.bc"],
    cmd = "mkdir -p polybench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/polybench-cpu-adi",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "polybench-atax",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["polybench/atax.bc"],
    cmd = "mkdir -p polybench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/polybench-cpu-atax",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "polybench-bicg",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["polybench/bicg.bc"],
    cmd = "mkdir -p polybench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/polybench-cpu-bicg",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "polybench-cholesky",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["polybench/cholesky.bc"],
    cmd = "mkdir -p polybench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/polybench-cpu-cholesky",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "polybench-doitgen",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["polybench/doitgen.bc"],
    cmd = "mkdir -p polybench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/polybench-cpu-doitgen",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "polybench-durbin",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["polybench/durbin.bc"],
    cmd = "mkdir -p polybench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/polybench-cpu-durbin",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "polybench-dynprog",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["polybench/dynprog.bc"],
    cmd = "mkdir -p polybench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/polybench-cpu-dynprog",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "polybench-fdtd-1",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["polybench/fdtd-1.bc"],
    cmd = "mkdir -p polybench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/polybench-cpu-fdtd-2d",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "polybench-fdtd-2",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["polybench/fdtd-2.bc"],
    cmd = "mkdir -p polybench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/polybench-cpu-fdtd-apml",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "polybench-gemm",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["polybench/gemm.bc"],
    cmd = "mkdir -p polybench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/polybench-cpu-gemm",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "polybench-gemver",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["polybench/gemver.bc"],
    cmd = "mkdir -p polybench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/polybench-cpu-gemver",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "polybench-gesummv",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["polybench/gesummv.bc"],
    cmd = "mkdir -p polybench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/polybench-cpu-gesummv",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "polybench-gramschmidt",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["polybench/gramschmidt.bc"],
    cmd = "mkdir -p polybench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/polybench-cpu-gramschmidt",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "polybench-jacobi-1d",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["polybench/jacobi-1d.bc"],
    cmd = "mkdir -p polybench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/polybench-cpu-jacobi-1d-imper",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "polybench-jacobi-2d",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["polybench/jacobi-2d.bc"],
    cmd = "mkdir -p polybench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/polybench-cpu-jacobi-2d-imper",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "polybench-lu",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["polybench/lu.bc"],
    cmd = "mkdir -p polybench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/polybench-cpu-lu",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "polybench-medley-floyd",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["polybench/medley-floyd.bc"],
    cmd = "mkdir -p polybench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/polybench-cpu-medley-floyd-warshall",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "polybench-medley-reg",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["polybench/medley-reg.bc"],
    cmd = "mkdir -p polybench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/polybench-cpu-medley-reg-detect",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "polybench-mvt",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["polybench/mvt.bc"],
    cmd = "mkdir -p polybench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/polybench-cpu-mvt",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "polybench-seidel",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["polybench/seidel.bc"],
    cmd = "mkdir -p polybench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/polybench-cpu-seidel-2d",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "polybench-symm",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["polybench/symm.bc"],
    cmd = "mkdir -p polybench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/polybench-cpu-symm",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "polybench-syr2k",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["polybench/syr2k.bc"],
    cmd = "mkdir -p polybench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/polybench-cpu-syr2k",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "polybench-syrk",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["polybench/syrk.bc"],
    cmd = "mkdir -p polybench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/polybench-cpu-syrk",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "polybench-trisolv",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["polybench/trisolv.bc"],
    cmd = "mkdir -p polybench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/polybench-cpu-trisolv",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)

genrule(
    name = "polybench-trmm",
    srcs = [
        "@ctuning-programs//:all",
        "@ctuning-programs//:readme",
    ],
    outs = ["polybench/trmm.bc"],
    cmd = "mkdir -p polybench && $(location :make_bitcode) $(location @llvm//10.0.0:clang) $(location @llvm//10.0.0:llvm-link) $@ $$(dirname $(location @ctuning-programs//:readme))/program/polybench-cpu-trmm",
    tools = [
        ":make_bitcode",
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
)
