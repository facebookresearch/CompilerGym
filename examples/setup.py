import distutils.util

import setuptools

with open("../VERSION") as f:
    version = f.read().strip()

setuptools.setup(
    name="compiler_gym_examples",
    version=version,
    description="Example code for CompilerGym",
    author="Facebook AI Research",
    url="https://github.com/facebookresearch/CompilerGym",
    license="MIT",
    packages=[
        "llvm_autotuning",
        "llvm_autotuning.autotuners",
    ],
    python_requires=">=3.8",
    platforms=[distutils.util.get_platform()],
    zip_safe=False,
)
