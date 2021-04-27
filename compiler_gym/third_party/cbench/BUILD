# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_binary")

genrule(
    name = "cbench_tar",
    srcs = [":cbench"],
    outs = ["llvm_bitcodes-10.0.0-cbench-v1.tar.bz2"],
    cmd = "tar cjfh $@ -C $(@D) cbench-v1",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "cbench",
    srcs = [
        ":adpcm",
        ":bitcount",
        ":blowfish",
        ":bzip2",
        ":crc32",
        ":dijkstra",
        ":ghostscript",
        ":gsm",
        ":ispell",
        ":jpeg-c",
        ":jpeg-d",
        ":lame",
        ":patricia",
        ":qsort",
        ":rijndael",
        ":sha",
        ":stringsearch",
        ":stringsearch2",
        ":susan",
        ":tiff2bw",
        ":tiff2rgba",
        ":tiffdither",
        ":tiffmedian",
    ],
    visibility = ["//visibility:public"],
)

genrule(
    name = "make_runtime_data",
    srcs = [
        "@cBench_consumer_tiff_data//file",
        "@cBench_office_data//file",
        "@cBench_telecom_data//file",
        "@cBench_consumer_jpeg_data//file",
        "@cBench_telecom_gsm_data//file",
        "@cBench_consumer_data//file",
        "@cBench_bzip2_data//file",
        "@cBench_network_patricia_data//file",
        "@cBench_network_dijkstra_data//file",
        "@cBench_automotive_susan_data//file",
        "@cBench_automotive_qsort_data//file",
    ] + glob(["runtime_data/**/*"]),
    outs = ["runtime_data"],
    cmd = (
        "cp -R $$(dirname $(location runtime_data/ghostscript/gs_init.ps))/../.. $(@D) &&" +
        "tar xzf $(location @cBench_consumer_tiff_data//file) -C $(@D)/runtime_data &&" +
        "tar xzf $(location @cBench_office_data//file) -C $(@D)/runtime_data &&" +
        "tar xzf $(location @cBench_telecom_data//file) -C $(@D)/runtime_data &&" +
        "tar xzf $(location @cBench_consumer_jpeg_data//file) -C $(@D)/runtime_data &&" +
        "tar xzf $(location @cBench_telecom_gsm_data//file) -C $(@D)/runtime_data &&" +
        "tar xzf $(location @cBench_consumer_data//file) -C $(@D)/runtime_data &&" +
        "tar xzf $(location @cBench_bzip2_data//file) -C $(@D)/runtime_data &&" +
        "tar xzf $(location @cBench_network_patricia_data//file) -C $(@D)/runtime_data &&" +
        "tar xzf $(location @cBench_network_dijkstra_data//file) -C $(@D)/runtime_data &&" +
        "tar xzf $(location @cBench_automotive_susan_data//file) -C $(@D)/runtime_data &&" +
        "tar xzf $(location @cBench_automotive_qsort_data//file) -C $(@D)/runtime_data"
    ),
)

filegroup(
    name = "benchmarks_list",
    srcs = ["benchmarks.txt"],
    visibility = ["//visibility:public"],
)

py_binary(
    name = "make_llvm_module",
    srcs = ["make_llvm_module.py"],
    deps = [
        "//compiler_gym/envs/llvm:llvm_benchmark",
    ],
)

genrule(
    name = "crc32",
    srcs = [
        "@cBench//:all",
        "@cBench//:readme",
    ],
    outs = ["cbench-v1/crc32.bc"],
    cmd = "mkdir -p $(@D) && $(location :make_llvm_module) $$(dirname $(location @cBench//:readme))/telecom_CRC32 $@",
    tools = [":make_llvm_module"],
    visibility = ["//visibility:public"],
)

genrule(
    name = "jpeg-c",
    srcs = [
        "@ctuning-ai//:all",
        "@ctuning-ai//:readme",
    ],
    outs = ["cbench-v1/jpeg-c.bc"],
    cmd = "mkdir -p $(@D) && $(location :make_llvm_module) $$(dirname $(location @ctuning-ai//:readme))/program/cbench-consumer-jpeg-c $@",
    tools = [
        ":make_llvm_module",
    ],
)

genrule(
    name = "jpeg-d",
    srcs = [
        "@ctuning-ai//:all",
        "@ctuning-ai//:readme",
    ],
    outs = ["cbench-v1/jpeg-d.bc"],
    cmd = "mkdir -p $(@D) && $(location :make_llvm_module) $$(dirname $(location @ctuning-ai//:readme))/program/cbench-consumer-jpeg-d $@",
    tools = [
        ":make_llvm_module",
    ],
)

genrule(
    name = "stringsearch2",
    srcs = [
        "@ctuning-ai//:all",
        "@ctuning-ai//:readme",
    ],
    outs = ["cbench-v1/stringsearch2.bc"],
    cmd = "mkdir -p $(@D) && $(location :make_llvm_module) $$(dirname $(location @ctuning-ai//:readme))/program/cbench-office-stringsearch2 $@",
    tools = [
        ":make_llvm_module",
    ],
)

genrule(
    name = "adpcm",
    srcs = [
        "@cBench//:all",
        "@cBench//:readme",
    ],
    outs = ["cbench-v1/adpcm.bc"],
    cmd = "mkdir -p $(@D) && $(location :make_llvm_module) $$(dirname $(location @cBench//:readme))/telecom_adpcm_c $@",
    tools = [":make_llvm_module"],
    visibility = ["//visibility:public"],
)

genrule(
    name = "bitcount",
    srcs = [
        "@cBench//:all",
        "@cBench//:readme",
    ],
    outs = ["cbench-v1/bitcount.bc"],
    cmd = "mkdir -p $(@D) && $(location :make_llvm_module) $$(dirname $(location @cBench//:readme))/automotive_bitcount $@",
    tools = [":make_llvm_module"],
)

genrule(
    name = "blowfish",
    srcs = [
        "@cBench//:all",
        "@cBench//:readme",
    ],
    outs = ["cbench-v1/blowfish.bc"],
    cmd = "mkdir -p $(@D) && $(location :make_llvm_module) $$(dirname $(location @cBench//:readme))/security_blowfish_d $@",
    tools = [":make_llvm_module"],
)

genrule(
    name = "bzip2",
    srcs = [
        "@cBench//:all",
        "@cBench//:readme",
    ],
    outs = ["cbench-v1/bzip2.bc"],
    cmd = "mkdir -p $(@D) && $(location :make_llvm_module) $$(dirname $(location @cBench//:readme))/bzip2d $@",
    tools = [":make_llvm_module"],
)

genrule(
    name = "dijkstra",
    srcs = [
        "@cBench//:all",
        "@cBench//:readme",
    ],
    outs = ["cbench-v1/dijkstra.bc"],
    cmd = "mkdir -p $(@D) && $(location :make_llvm_module) $$(dirname $(location @cBench//:readme))/network_dijkstra $@",
    tools = [":make_llvm_module"],
)

genrule(
    name = "ghostscript",
    srcs = [
        "@cBench//:all",
        "@cBench//:readme",
        "cBench-ghostscript-std.h.patch",
        "cBench-ghostscript-idebug.c.patch",
    ],
    outs = ["cbench-v1/ghostscript.bc"],
    cmd = (
        "mkdir -p $(@D) && rsync -rL $$(dirname $(location @cBench//:readme))/office_ghostscript/ $(@D)/office_ghostscript_src/ &&" +
        "patch --quiet --forward $(@D)/office_ghostscript_src/src/idebug.c < $(location cBench-ghostscript-idebug.c.patch);" +
        "patch --quiet --forward $(@D)/office_ghostscript_src/src/std.h < $(location cBench-ghostscript-std.h.patch);" +
        "$(location :make_llvm_module) $(@D)/office_ghostscript_src $@"
    ),
    tools = [":make_llvm_module"],
)

genrule(
    name = "gsm",
    srcs = [
        "@cBench//:all",
        "@cBench//:readme",
        "cBench-gsm-add.c.patch",
    ],
    outs = ["cbench-v1/gsm.bc"],
    cmd = (
        "mkdir -p $(@D) &&rsync -rL $$(dirname $(location @cBench//:readme))/telecom_gsm/ $(@D)/telecom_gsm_src/ &&" +
        "patch --quiet --forward $(@D)/telecom_gsm_src/src/add.c < $(location cBench-gsm-add.c.patch);" +
        "$(location :make_llvm_module) $(@D)/telecom_gsm_src $@ -DSASR -DSTUPID_COMPILER -DNeedFunctionPrototypes=1"
    ),
    tools = [":make_llvm_module"],
)

genrule(
    name = "ispell",
    srcs = [
        "@cBench//:all",
        "@cBench//:readme",
        "cBench-ispell-correct.c.patch",
    ],
    outs = ["cbench-v1/ispell.bc"],
    cmd = (
        "mkdir -p $(@D) &&rsync -rL $$(dirname $(location @cBench//:readme))/office_ispell/ $(@D)/office_ispell_src/ &&" +
        "patch --quiet --forward $(@D)/office_ispell_src/src/correct.c < $(location cBench-ispell-correct.c.patch);" +
        "$(location :make_llvm_module) $(@D)/office_ispell_src $@"
    ),
    tools = [":make_llvm_module"],
)

genrule(
    name = "lame",
    srcs = [
        "@cBench//:all",
        "@cBench//:readme",
    ],
    outs = ["cbench-v1/lame.bc"],
    cmd = "mkdir -p $(@D) && $(location :make_llvm_module) $$(dirname $(location @cBench//:readme))/consumer_lame $@ -DLAMESNDFILE -DHAVEMPGLIB -DLAMEPARSE",
    tools = [":make_llvm_module"],
)

genrule(
    name = "patricia",
    srcs = [
        "@cBench//:all",
        "@cBench//:readme",
    ],
    outs = ["cbench-v1/patricia.bc"],
    cmd = (
        "mkdir -p $(@D) && $(location :make_llvm_module) $$(dirname $(location @cBench//:readme))/network_patricia/ $@"
    ),
    tools = [":make_llvm_module"],
)

genrule(
    name = "qsort",
    srcs = [
        "@cBench//:all",
        "@cBench//:readme",
    ],
    outs = ["cbench-v1/qsort.bc"],
    cmd = "mkdir -p $(@D) && $(location :make_llvm_module) $$(dirname $(location @cBench//:readme))/automotive_qsort1 $@",
    tools = [":make_llvm_module"],
)

genrule(
    name = "rijndael",
    srcs = [
        "@cBench//:all",
        "@cBench//:readme",
    ],
    outs = ["cbench-v1/rijndael.bc"],
    cmd = "mkdir -p $(@D) && $(location :make_llvm_module) $$(dirname $(location @cBench//:readme))/security_rijndael_d $@",
    tools = [":make_llvm_module"],
)

genrule(
    name = "sha",
    srcs = [
        "@cBench//:all",
        "@cBench//:readme",
    ],
    outs = ["cbench-v1/sha.bc"],
    cmd = "mkdir -p $(@D) && $(location :make_llvm_module) $$(dirname $(location @cBench//:readme))/security_sha $@",
    tools = [":make_llvm_module"],
)

genrule(
    name = "stringsearch",
    srcs = [
        "@cBench//:all",
        "@cBench//:readme",
    ],
    outs = ["cbench-v1/stringsearch.bc"],
    cmd = "mkdir -p $(@D) && $(location :make_llvm_module) $$(dirname $(location @cBench//:readme))/office_stringsearch1 $@",
    tools = [":make_llvm_module"],
)

genrule(
    name = "susan",
    srcs = [
        "@cBench//:all",
        "@cBench//:readme",
    ],
    outs = ["cbench-v1/susan.bc"],
    cmd = "mkdir -p $(@D) && $(location :make_llvm_module) $$(dirname $(location @cBench//:readme))/automotive_susan_c $@",
    tools = [":make_llvm_module"],
)

genrule(
    name = "tiff2bw",
    srcs = [
        "@cBench//:all",
        "@cBench//:readme",
    ],
    outs = ["cbench-v1/tiff2bw.bc"],
    cmd = "mkdir -p $(@D) && $(location :make_llvm_module) $$(dirname $(location @cBench//:readme))/consumer_tiff2bw $@",
    tools = [":make_llvm_module"],
)

genrule(
    name = "tiff2rgba",
    srcs = [
        "@cBench//:all",
        "@cBench//:readme",
    ],
    outs = ["cbench-v1/tiff2rgba.bc"],
    cmd = "mkdir -p $(@D) && $(location :make_llvm_module) $$(dirname $(location @cBench//:readme))/consumer_tiff2rgba $@",
    tools = [":make_llvm_module"],
)

genrule(
    name = "tiffdither",
    srcs = [
        "@cBench//:all",
        "@cBench//:readme",
    ],
    outs = ["cbench-v1/tiffdither.bc"],
    cmd = "mkdir -p $(@D) && $(location :make_llvm_module) $$(dirname $(location @cBench//:readme))/consumer_tiffdither $@",
    tools = [":make_llvm_module"],
)

genrule(
    name = "tiffmedian",
    srcs = [
        "@cBench//:all",
        "@cBench//:readme",
    ],
    outs = ["cbench-v1/tiffmedian.bc"],
    cmd = "mkdir -p $(@D) && $(location :make_llvm_module) $$(dirname $(location @cBench//:readme))/consumer_tiffmedian $@",
    tools = [":make_llvm_module"],
    visibility = ["//visibility:public"],
)
