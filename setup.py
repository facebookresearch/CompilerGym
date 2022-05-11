#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import distutils.command.build
import distutils.util
import fnmatch
import glob
import io
import os
import sys
from pathlib import Path

import setuptools
from setuptools.command.build_py import build_py as build_py_orig
from setuptools.dist import Distribution

argparser = argparse.ArgumentParser(add_help=False)
argparser.add_argument(
    "--package-dir",
    help="Source directory of package files.",
    default="bazel-bin/package.runfiles/CompilerGym",
)
argparser.add_argument(
    "--get-wheel-filename",
    action="store_true",
    help="Print only output filename without building it.",
)
argparser.add_argument(
    "--build-dir",
    help="Path to build dir. This is where this script copies files from the source before making the wheel package.",
    default="build",
)
args, unknown = argparser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown

sys.path.insert(0, str((Path(args.package_dir) / "compiler_gym").absolute()))
import config  # noqa: E402

with open("VERSION") as f:
    version = f.read().strip()
with open("README.md") as f:
    # Force UTF-8 file encoding to support non-ascii characters in the readme.
    with io.open("README.md", encoding="utf-8") as f:
        long_description = f.read()
with open("compiler_gym/requirements.txt") as f:
    requirements = [ln.split("#")[0].rstrip() for ln in f.readlines()]

# When building a bdist_wheel we need to set the appropriate tags: this package
# includes compiled binaries, and does not include compiled python extensions.
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False

        def get_tag(self):
            python, abi, plat = _bdist_wheel.get_tag(self)
            python, abi = "py3", "none"
            return python, abi, plat

except ImportError:
    bdist_wheel = None


class build(distutils.command.build.build):
    def initialize_options(self):
        distutils.command.build.build.initialize_options(self)
        self.build_base = args.build_dir


# Add files that should be excluded from the package.
# The argument exclude_package_data of setuptools.setup(...)
# does not work with py files. They have to be excluded here.
excluded = [
    str(Path(args.package_dir) / "compiler_gym/envs/llvm/make_specs.py"),
    str(Path(args.package_dir) / "compiler_gym/bin/random_eval.py"),
]


class build_py(build_py_orig):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        res = [
            (pkg, mod, file)
            for (pkg, mod, file) in modules
            if not any(fnmatch.fnmatchcase(file, pat=pattern) for pattern in excluded)
        ]
        return res


def wheel_filename(**kwargs):
    # create a fake distribution from arguments
    dist = Distribution(attrs=kwargs)
    # finalize bdist_wheel command
    bdist_wheel_cmd = dist.get_command_obj("bdist_wheel")
    bdist_wheel_cmd.ensure_finalized()
    # assemble wheel file name
    distname = bdist_wheel_cmd.wheel_dist_name
    tag = "-".join(bdist_wheel_cmd.get_tag())
    return f"{distname}-{tag}.whl"


setup_kwargs = {
    "name": "compiler_gym",
    "version": version,
    "description": "Reinforcement learning environments for compiler research",
    "author": "Facebook AI Research",
    "long_description": long_description,
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/facebookresearch/CompilerGym",
    "license": "MIT",
    "packages": [
        "compiler_gym.bin",
        "compiler_gym.datasets",
        "compiler_gym.envs.gcc.datasets",
        "compiler_gym.envs.gcc.service",
        "compiler_gym.envs.gcc",
        "compiler_gym.envs.cgra.datasets",
        "compiler_gym.envs.cgra.service",
        "compiler_gym.envs.cgra",
        "compiler_gym.envs.loop_tool",
        "compiler_gym.envs.loop_tool.service",
        "compiler_gym.envs",
        "compiler_gym.envs",
        "compiler_gym.errors",
        "compiler_gym.leaderboard",
        "compiler_gym.service.proto",
        "compiler_gym.service.runtime",
        "compiler_gym.service",
        "compiler_gym.spaces",
        "compiler_gym.third_party.inst2vec",
        "compiler_gym.third_party",
        "compiler_gym.util.flags",
        "compiler_gym.util",
        "compiler_gym.views",
        "compiler_gym.wrappers",
        "compiler_gym",
    ],
    "package_dir": {
        "": args.package_dir,
    },
    "package_data": {
        "compiler_gym": [
            "envs/gcc/service/compiler_gym-gcc-service",
            # "envs/cgra/service/compiler_gym-cgra-service",
            "envs/cgra/service/*",
            "envs/loop_tool/service/compiler_gym-loop_tool-service",
            "third_party/csmith/csmith/bin/csmith",
            "third_party/csmith/csmith/include/csmith-2.3.0/*.h",
            "third_party/inst2vec/*.pickle",
        ]
    },
    "install_requires": requirements,
    "include_package_data": True,
    "python_requires": ">=3.6",
    "classifiers": [
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Compilers",
    ],
    "cmdclass": {"bdist_wheel": bdist_wheel, "build": build, "build_py": build_py},
    "platforms": [distutils.util.get_platform()],
    "zip_safe": False,
}

if config.enable_llvm_env:
    setup_kwargs["packages"].extend(
        [
            "compiler_gym.envs.llvm.datasets",
            "compiler_gym.envs.llvm.service",
            "compiler_gym.envs.llvm",
            "compiler_gym.third_party.llvm",
            "compiler_gym.third_party.autophase",
        ]
    )
    setup_kwargs["package_data"]["compiler_gym"].extend(
        [
            "envs/llvm/service/compiler_gym-llvm-service",
            "envs/llvm/service/compute_observation",
            "envs/llvm/service/libLLVMPolly.so",
            "third_party/cbench/benchmarks.txt",
            "third_party/cbench/cbench-v*/crc32.bc",
        ]
    )

if config.enable_mlir_env:
    setup_kwargs["packages"].extend(
        [
            "compiler_gym.envs.mlir.datasets",
            "compiler_gym.envs.mlir.service",
            "compiler_gym.envs.mlir",
        ]
    )
    setup_kwargs["package_data"]["compiler_gym"].extend(
        ["envs/mlir/service/compiler_gym-mlir-service"]
    )
    original_cwd = os.getcwd()
    try:
        os.chdir(os.path.join(args.package_dir, "compiler_gym"))
        setup_kwargs["package_data"]["compiler_gym"].extend(
            glob.glob("envs/mlir/service/llvm/**", recursive=True)
        )
        setup_kwargs["package_data"]["compiler_gym"].extend(
            glob.glob("envs/mlir/service/google_benchmark/**", recursive=True)
        )
    finally:
        os.chdir(original_cwd)

if args.get_wheel_filename:
    # Instead of generating the wheel file,
    # print its filename.
    file_name = wheel_filename(**setup_kwargs)
    sys.stdout.write(file_name)
else:
    setuptools.setup(**setup_kwargs)
