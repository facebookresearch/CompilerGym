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
        "mkdir -p $(@D) && rsync -rL $$(dirname $(location @cBench//:readme))/telecom_gsm/ $(@D)/telecom_gsm_src/ &&" +
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
        "mkdir -p $(@D) && rsync -rL $$(dirname $(location @cBench//:readme))/office_ispell/ $(@D)/office_ispell_src/ &&" +
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

# Definition of the runtime data archive that we host at:
# https://dl.fbaipublicfiles.com/compiler_gym/cBench-v0-runtime-data.tar.bz2
#
# genrule(
#     name = "make_runtime_data",
#     srcs = [
#         "@cBench_consumer_tiff_data//file",
#         "@cBench_office_data//file",
#         "@cBench_telecom_data//file",
#         "@cBench_consumer_jpeg_data//file",
#         "@cBench_telecom_gsm_data//file",
#         "@cBench_consumer_data//file",
#         "@cBench_bzip2_data//file",
#         "@cBench_network_patricia_data//file",
#         "@cBench_network_dijkstra_data//file",
#         "@cBench_automotive_susan_data//file",
#         "@cBench_automotive_qsort_data//file",
#     ] + glob(["runtime_data/**/*"]),
#     outs = ["runtime_data"],
#     cmd = (
#         "cp -R $$(dirname $(location runtime_data/ghostscript/gs_init.ps))/../.. $(@D) &&" +
#         "tar xzf $(location @cBench_consumer_tiff_data//file) -C $(@D)/runtime_data &&" +
#         "tar xzf $(location @cBench_office_data//file) -C $(@D)/runtime_data &&" +
#         "tar xzf $(location @cBench_telecom_data//file) -C $(@D)/runtime_data &&" +
#         "tar xzf $(location @cBench_consumer_jpeg_data//file) -C $(@D)/runtime_data &&" +
#         "tar xzf $(location @cBench_telecom_gsm_data//file) -C $(@D)/runtime_data &&" +
#         "tar xzf $(location @cBench_consumer_data//file) -C $(@D)/runtime_data &&" +
#         "tar xzf $(location @cBench_bzip2_data//file) -C $(@D)/runtime_data &&" +
#         "tar xzf $(location @cBench_network_patricia_data//file) -C $(@D)/runtime_data &&" +
#         "tar xzf $(location @cBench_network_dijkstra_data//file) -C $(@D)/runtime_data &&" +
#         "tar xzf $(location @cBench_automotive_susan_data//file) -C $(@D)/runtime_data &&" +
#         "tar xzf $(location @cBench_automotive_qsort_data//file) -C $(@D)/runtime_data"
#     ),
# )
#
###############################################################################
# This depends on the following WORKSPACE targets:
#
# http_file(
#     name = "cBench_consumer_tiff_data",
#     sha256 = "779abb7b7fee8733313e462e6066c16375e9209a9f7ff692fd06c7598946939a",
#     urls = ["https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_consumer_tiff_data.tar.gz"],
# )
#
# http_file(
#     name = "cBench_office_data",
#     sha256 = "cfa09cd37cb93aba57415033905dc6308653c7b833feba5a25067bfb62999f32",
#     urls = ["https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_office_data.tar.gz"],
# )
#
# http_file(
#     name = "cBench_telecom_data",
#     sha256 = "e5cb6663beefe32fd12f90c8f533f8e1bce2f05ee4e3836efb5556d5e1089df0",
#     urls = ["https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_telecom_data.tar.gz"],
# )
#
# http_file(
#     name = "cBench_consumer_jpeg_data",
#     sha256 = "bec5ffc15cd2f952d9a786f3cd31d90955c318a5e4f69c5ba472f79d5a3e8f0b",
#     urls = ["https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_consumer_jpeg_data.tar.gz"],
# )
#
# http_file(
#     name = "cBench_telecom_gsm_data",
#     sha256 = "52545d3a0ce15021131c62d96d3a3d7e6670e2d6c34226ac9a3d5191a1ee214a",
#     urls = ["https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_telecom_gsm_data.tar.gz"],
# )
#
# http_file(
#     name = "cBench_consumer_data",
#     sha256 = "a4d40344af3022bfd7b4c6fcf6d59d598825b07d9e37769dbf1b3effa39aa445",
#     urls = ["https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_consumer_data.tar.gz"],
# )
#
# http_file(
#     name = "cBench_bzip2_data",
#     sha256 = "46e5760eeef77e6b0c273af92de971bc45f33a59e0efc183073d9aa6b716c302",
#     urls = ["https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_bzip2_data.tar.gz"],
# )
#
# http_file(
#     name = "cBench_network_patricia_data",
#     sha256 = "72dae0e670d93ef929e50aca7a138463e0915502281ccafe793e378cb2a85dfb",
#     urls = ["https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_network_patricia_data.tar.gz"],
# )
#
# http_file(
#     name = "cBench_network_dijkstra_data",
#     sha256 = "41c13f59cdfbc772081cd941f499b030370bc570fc2ba60a5c4b7194bc36ca5f",
#     urls = ["https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_network_dijkstra_data.tar.gz"],
# )
#
# http_file(
#     name = "cBench_automotive_susan_data",
#     sha256 = "df56e1e44ccc560072381cdb001d770003ac74f92593dd5dbdfdd4ff9332a8e6",
#     urls = ["https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_automotive_susan_data.tar.gz"],
# )
#
# http_file(
#     name = "cBench_automotive_qsort_data",
#     sha256 = "510b4225021408ac190f6f793e7d7171d3553c9916cfa8b2fb4ace005105e768",
#     urls = ["https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_automotive_qsort_data.tar.gz"],
# )
