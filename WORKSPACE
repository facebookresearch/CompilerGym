workspace(name = "CompilerGym")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

# === Google test ===

http_archive(
    name = "gtest",
    sha256 = "9dc9157a9a1551ec7a7e43daea9a694a0bb5fb8bec81235d8a1e6ef64c716dcb",
    strip_prefix = "googletest-release-1.10.0",
    urls = [
        "https://github.com/google/googletest/archive/release-1.10.0.tar.gz",
    ],
)

# === Google flags ===

http_archive(
    name = "gflags",
    sha256 = "34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf",
    strip_prefix = "gflags-2.2.2",
    urls = ["https://github.com/gflags/gflags/archive/v2.2.2.tar.gz"],
)

# === Google logging ===

http_archive(
    name = "glog",
    sha256 = "f28359aeba12f30d73d9e4711ef356dc842886968112162bc73002645139c39c",
    strip_prefix = "glog-0.4.0",
    urls = ["https://github.com/google/glog/archive/v0.4.0.tar.gz"],
)

# === LLVM ===

http_archive(
    name = "llvm",
    sha256 = "47744dbbbe01f107682503348cb570c771152e44d9c75b7f45cf17dc2711ee7d",
    strip_prefix = "bazel_llvm-9481d3c85247bbc284d8283aa5b6b8b517301262",
    urls = ["https://github.com/ChrisCummins/bazel_llvm/archive/9481d3c85247bbc284d8283aa5b6b8b517301262.tar.gz"],
)

load("@llvm//tools/bzl:deps.bzl", "llvm_deps")

llvm_deps()

# === Building CMake projects ===

http_archive(
    name = "rules_foreign_cc",
    sha256 = "9dcf6f79c37e2e71a02ebcf21eea29f39099f9a779ef81674b1010acd744abba",
    strip_prefix = "rules_foreign_cc-e285764b78cc91e97cfead87f48d06b9c4d83a81",
    url = "https://github.com/ChrisCummins/rules_foreign_cc/archive/e285764b78cc91e97cfead87f48d06b9c4d83a81.zip",
)

load("@rules_foreign_cc//:workspace_definitions.bzl", "rules_foreign_cc_dependencies")

rules_foreign_cc_dependencies([])

# === Python rules ===

http_archive(
    name = "rules_python",
    sha256 = "b5668cde8bb6e3515057ef465a35ad712214962f0b3a314e551204266c7be90c",
    strip_prefix = "rules_python-0.0.2",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.0.2/rules_python-0.0.2.tar.gz",
)

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

# === Protocol buffers ===

http_archive(
    name = "rules_proto",
    sha256 = "66bfdf8782796239d3875d37e7de19b1d94301e8972b3cbd2446b332429b4df1",
    strip_prefix = "rules_proto-4.0.0",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_proto/archive/refs/tags/4.0.0.tar.gz",
        "https://github.com/bazelbuild/rules_proto/archive/refs/tags/4.0.0.tar.gz",
    ],
)

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")

rules_proto_dependencies()

rules_proto_toolchains()

# === GRPC ===

# Version should be kept in step with compiler_gym/requirements.txt.
http_archive(
    name = "com_github_grpc_grpc",
    sha256 = "cdeb805385fba23242bf87073e68d590c446751e09089f26e5e0b3f655b0f089",
    strip_prefix = "grpc-1.49.2",
    urls = [
        "https://github.com/grpc/grpc/archive/v1.49.2.tar.gz",
    ],
)

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")

grpc_extra_deps()

# === C++ enum trickery ===
# https://github.com/Neargye/magic_enum

http_archive(
    name = "magic_enum",
    build_file_content = """
cc_library(
    name = "magic_enum",
    hdrs = ["include/magic_enum.hpp"],
    includes = ["include"],
    visibility = ["//visibility:public"],
)
""",
    sha256 = "c7e96ad8bd260a831a59d2779ab8ceb9f1a056da177fdb5a3fec538cfaac9d52",
    strip_prefix = "magic_enum-6e932ef66dbe054e039d4dba77a41a12f9f52e0c",
    urls = ["https://github.com/Neargye/magic_enum/archive/6e932ef66dbe054e039d4dba77a41a12f9f52e0c.tar.gz"],
)

# === ctuning-programs ===
# https://github.com/ChrisCummins/ctuning-programs

http_archive(
    name = "ctuning-programs",
    build_file_content = """
filegroup(
    name = "all",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "readme",
    srcs = ["README.md"],
    visibility = ["//visibility:public"],
)
""",
    sha256 = "5e14a49f87c70999a082cb5cf19b780d0b56186f63356f8f994dd9ffc79ec6f3",
    strip_prefix = "ctuning-programs-c3c126fcb400f3a14b69b152f15d15eae78ef908",
    urls = ["https://github.com/ChrisCummins/ctuning-programs/archive/c3c126fcb400f3a14b69b152f15d15eae78ef908.tar.gz"],
)

# === cBench ===
# https://ctuning.org/wiki/index.php/CTools:CBench

all_content = """filegroup(name = "all", srcs = glob(["**"]), visibility = ["//visibility:public"])"""

http_archive(
    name = "cBench",
    build_file_content = """
filegroup(
    name = "all",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "readme",
    srcs = ["README-cBench-V1.1.txt"],
    visibility = ["//visibility:public"],
)
""",
    sha256 = "8908d742f5223f09f9a4d10f7e06bc805a0c1694aa70974d2aae91ab627b51e6",
    urls = [
        "https://dl.fbaipublicfiles.com/compiler_gym/cBench_V1.1.tar.gz",
        "https://downloads.sourceforge.net/project/cbenchmark/cBench/V1.1/cBench_V1.1.tar.gz",
    ],
)

http_archive(
    name = "ctuning-ai",
    build_file = "//:compiler_gym/third_party/ctuning-ai.BUILD",
    sha256 = "a82c13733696c46b5201c614fcf7229c3a74a83ce485cab2fbf17309b7564f9c",
    strip_prefix = "ck-mlops-406738ad6d1fb2c1da9daa2c09d26fccab4e0938",
    urls = ["https://github.com/ChrisCummins/ck-mlops/archive/406738ad6d1fb2c1da9daa2c09d26fccab4e0938.tar.gz"],
)

# Datasets.

http_file(
    name = "cBench_consumer_tiff_data",
    sha256 = "779abb7b7fee8733313e462e6066c16375e9209a9f7ff692fd06c7598946939a",
    urls = ["https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_consumer_tiff_data.tar.gz"],
)

http_file(
    name = "cBench_office_data",
    sha256 = "cfa09cd37cb93aba57415033905dc6308653c7b833feba5a25067bfb62999f32",
    urls = ["https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_office_data.tar.gz"],
)

http_file(
    name = "cBench_telecom_data",
    sha256 = "e5cb6663beefe32fd12f90c8f533f8e1bce2f05ee4e3836efb5556d5e1089df0",
    urls = ["https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_telecom_data.tar.gz"],
)

http_file(
    name = "cBench_consumer_jpeg_data",
    sha256 = "bec5ffc15cd2f952d9a786f3cd31d90955c318a5e4f69c5ba472f79d5a3e8f0b",
    urls = ["https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_consumer_jpeg_data.tar.gz"],
)

http_file(
    name = "cBench_telecom_gsm_data",
    sha256 = "52545d3a0ce15021131c62d96d3a3d7e6670e2d6c34226ac9a3d5191a1ee214a",
    urls = ["https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_telecom_gsm_data.tar.gz"],
)

http_file(
    name = "cBench_consumer_data",
    sha256 = "a4d40344af3022bfd7b4c6fcf6d59d598825b07d9e37769dbf1b3effa39aa445",
    urls = ["https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_consumer_data.tar.gz"],
)

http_file(
    name = "cBench_bzip2_data",
    sha256 = "46e5760eeef77e6b0c273af92de971bc45f33a59e0efc183073d9aa6b716c302",
    urls = ["https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_bzip2_data.tar.gz"],
)

http_file(
    name = "cBench_network_patricia_data",
    sha256 = "72dae0e670d93ef929e50aca7a138463e0915502281ccafe793e378cb2a85dfb",
    urls = ["https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_network_patricia_data.tar.gz"],
)

http_file(
    name = "cBench_network_dijkstra_data",
    sha256 = "41c13f59cdfbc772081cd941f499b030370bc570fc2ba60a5c4b7194bc36ca5f",
    urls = ["https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_network_dijkstra_data.tar.gz"],
)

http_file(
    name = "cBench_automotive_susan_data",
    sha256 = "df56e1e44ccc560072381cdb001d770003ac74f92593dd5dbdfdd4ff9332a8e6",
    urls = ["https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_automotive_susan_data.tar.gz"],
)

http_file(
    name = "cBench_automotive_qsort_data",
    sha256 = "510b4225021408ac190f6f793e7d7171d3553c9916cfa8b2fb4ace005105e768",
    urls = ["https://downloads.sourceforge.net/project/cbenchmark/cDatasets/V1.1/cDatasets_V1.1_automotive_qsort_data.tar.gz"],
)

# === C++ cpuinfo ===

http_archive(
    name = "org_pytorch_cpuinfo",
    sha256 = "0936848904943381b2c01321101614776e43d583840ee0f3ceeea1e3fb7405f7",
    strip_prefix = "cpuinfo-de2fa78ebb431db98489e78603e4f77c1f6c5c57",
    urls = ["https://github.com/pytorch/cpuinfo/archive/de2fa78ebb431db98489e78603e4f77c1f6c5c57.tar.gz"],
)

# === Csmith ===
# https://embed.cs.utah.edu/csmith/

http_archive(
    name = "csmith",
    build_file_content = all_content,
    sha256 = "9d024a6b202f6a1b9e01351218a85888c06b67b837fe4c6f8ef5bd522fae098c",
    strip_prefix = "csmith-csmith-2.3.0",
    urls = [
        "https://github.com/ChrisCummins/csmith/archive/refs/tags/csmith-2.3.0.tar.gz",
        "https://github.com/csmith-project/csmith/archive/refs/tags/csmith-2.3.0.tar.gz",
    ],
)

# === DeepDataFlow ===
# https://zenodo.org/record/4122437

http_archive(
    name = "DeepDataFlow",
    build_file = "//:third_party/DeepDataFlow/DeepDataFlow.BUILD",
    sha256 = "ea6accbeb005889db3ecaae99403933c1008e0f2f4adc3c4afae3d7665c54004",
    urls = ["https://zenodo.org/record/4122437/files/llvm_bc_20.06.01.tar.bz2?download=1"],
)

# === A modern C++ formatting library ===
# https://fmt.dev

http_archive(
    name = "fmt",
    build_file_content = """
cc_library(
    name = "fmt",
    srcs = glob(["src/*.cc"]),
    hdrs = glob(["include/fmt/*.h"]),
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)
""",
    sha256 = "1cafc80701b746085dddf41bd9193e6d35089e1c6ec1940e037fcb9c98f62365",
    strip_prefix = "fmt-6.1.2",
    urls = ["https://github.com/fmtlib/fmt/archive/6.1.2.tar.gz"],
)

# === Boost ===
# https://github.com/nelhage/rules_boost

http_archive(
    name = "com_github_nelhage_rules_boost",
    sha256 = "4031539fe0af832c6b6ed6974d820d350299a291ba7337d6c599d4854e47ed88",
    strip_prefix = "rules_boost-4ee400beca08f524e7ea3be3ca41cce34454272f",
    urls = ["https://github.com/nelhage/rules_boost/archive/4ee400beca08f524e7ea3be3ca41cce34454272f.tar.gz"],
)

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")

boost_deps()

# === ProGraML ===
# https://github.com/ChrisCummins/ProGraML

http_archive(
    name = "programl",
    sha256 = "dbe16a96963c31620bcd7fb60cad7cab76f67e413c70f11ea38347e389f9e1de",
    strip_prefix = "ProGraML-2b18ea3f1b7ba5668c4f594c555ccb52d735d4bc",
    urls = ["https://github.com/ChrisCummins/ProGraML/archive/2b18ea3f1b7ba5668c4f594c555ccb52d735d4bc.tar.gz"],
)

load("@programl//tools:bzl/deps.bzl", "programl_deps")

programl_deps()

# === IR2Vec ===
# https://github.com/IITH-Compilers/IR2Vec

http_archive(
    name = "ir2vec",
    build_file_content = """
genrule(
    name = "version",
    outs = ["version.h"],
    cmd = "echo '#define IR2VEC_VERSION \\"1\\"' > $@",
)

cc_library(
    name = "ir2vec",
    srcs = glob(["src/*.cpp"]) + [":version.h"],
    hdrs = glob(["src/include/*.h"]),
    copts = ["-Iexternal/ir2vec/src/include"],
    strip_include_prefix = "src/include",
    visibility = ["//visibility:public"],
    deps = [
        "@eigen//:eigen",
        "@llvm//10.0.0",
    ],
)
""",
    sha256 = "f6c5af059840889e584c13331fabc6a469c40cdf0e44b3284e7db4fe9093289c",
    strip_prefix = "IR2Vec-828e50584b9c8bc305208e22d2cca272bdb1ab64",
    urls = ["https://github.com/ChrisCummins/IR2Vec/archive/828e50584b9c8bc305208e22d2cca272bdb1ab64.tar.gz"],
)

# === Eigen ===
# https://eigen.tuxfamily.org/index.php?title=Main_Page

http_archive(
    name = "eigen",
    build_file_content = """
cc_library(
    name = "eigen",
    hdrs = glob(["Eigen/**/*"]),
    visibility = ["//visibility:public"],
)
""",
    sha256 = "d56fbad95abf993f8af608484729e3d87ef611dd85b3380a8bad1d5cbc373a57",
    strip_prefix = "eigen-3.3.7",
    urls = ["https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz"],
)
