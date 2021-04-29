#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import distutils.util
import io

import setuptools

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

setuptools.setup(
    name="compiler_gym",
    version=version,
    description="Reinforcement learning environments for compiler research",
    author="Facebook AI Research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/CompilerGym",
    license="MIT",
    packages=[
        "compiler_gym.bin",
        "compiler_gym.datasets",
        "compiler_gym.envs.llvm.datasets",
        "compiler_gym.envs.llvm.service.passes",
        "compiler_gym.envs.llvm.service",
        "compiler_gym.envs.llvm",
        "compiler_gym.envs.llvm",
        "compiler_gym.envs",
        "compiler_gym.envs",
        "compiler_gym.leaderboard",
        "compiler_gym.service.proto",
        "compiler_gym.service",
        "compiler_gym.spaces",
        "compiler_gym.third_party.autophase",
        "compiler_gym.third_party.inst2vec",
        "compiler_gym.third_party.llvm",
        "compiler_gym.third_party",
        "compiler_gym.util.flags",
        "compiler_gym.util",
        "compiler_gym.views",
        "compiler_gym",
    ],
    package_dir={
        "": "bazel-bin/package.runfiles/CompilerGym",
    },
    package_data={
        "compiler_gym": [
            "envs/llvm/service/compiler_gym-llvm-service",
            "envs/llvm/service/libLLVMPolly.so",
            "envs/llvm/service/passes/*.txt",
            "third_party/cbench/benchmarks.txt",
            "third_party/cbench/cbench-v*/*",
            "third_party/inst2vec/*.pickle",
        ]
    },
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.6",
    classifiers=[
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
    cmdclass={"bdist_wheel": bdist_wheel},
    platforms=[distutils.util.get_platform()],
    zip_safe=False,
)
