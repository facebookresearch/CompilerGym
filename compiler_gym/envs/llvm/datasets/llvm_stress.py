# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import subprocess
from pathlib import Path
from typing import Iterable, Optional

from compiler_gym.datasets import Benchmark, Dataset
from compiler_gym.third_party import llvm

# The maximum value for the --seed argument to llvm-stress.
UINT_MAX = (2 << 32) - 1


class LlvmStressDataset(Dataset):
    """
    A dataset which uses llvm-stress to generate programs.

    `llvm-stress <https://llvm.org/docs/CommandGuide/llvm-stress.html>`_ is a
    tool for generating random LLVM-IR files.

    There are 2^32 - 1 random programs.

    Iteration order is stable.
    """

    def __init__(self, site_data_base: Path, sort_order: int = 0):
        super().__init__(
            name="generator://llvm-stress-v0",
            description="Randomly generated LLVM-IR",
            long_description_url="https://llvm.org/docs/CommandGuide/llvm-stress.html",
            license="Apache License v2.0 with LLVM Exceptions",
            site_data_base=site_data_base,
            sort_order=sort_order,
        )

    @property
    def n(self) -> int:
        # Actually (2 << 32) - 1, but practically infinite for all intents and purposes.
        return -1

    def benchmark_uris(self) -> Iterable[str]:
        return (f"generator://llvm-stress-v0/{i}" for i in range(UINT_MAX))

    def benchmark(self, uri: Optional[str] = None):
        if uri is None:
            seed = self.random.integers(UINT_MAX)
        else:
            seed = int(uri.split("/")[-1])

        # Run llvm-stress with the given seed and pipe the output to llvm-as to
        # assemble a bitcode.
        llvm_stress = subprocess.Popen(
            [str(llvm.llvm_stress_path()), f"--seed={seed}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        llvm_as = subprocess.Popen(
            [str(llvm.llvm_as_path()), "-"],
            stdin=llvm_stress.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        stdout, _ = llvm_as.communicate(timeout=60)
        if llvm_stress.returncode or llvm_as.returncode:
            raise OSError("Failed to generate benchmark")

        return Benchmark.from_file_contents(
            f"generator://llvm-stress-v0/{seed}", stdout
        )
