# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module defines and registers the example gym environments."""
import subprocess
from pathlib import Path
from typing import Iterable

from compiler_gym.datasets import Benchmark, Dataset
from compiler_gym.datasets.uri import BenchmarkUri
from compiler_gym.envs.llvm.llvm_benchmark import get_system_includes
from compiler_gym.spaces import Reward
from compiler_gym.third_party import llvm
from compiler_gym.util.registration import register
from compiler_gym.util.runfiles_path import runfiles_path, site_data_path

UNROLLING_PY_SERVICE_BINARY: Path = runfiles_path(
    "examples/example_unrolling_service/service_py/example-unrolling-service-py"
)

BENCHMARKS_PATH: Path = runfiles_path("examples/example_unrolling_service/benchmarks")

NEURO_VECTORIZER_HEADER: Path = runfiles_path(
    "compiler_gym/third_party/neuro-vectorizer/header.h"
)


class RuntimeReward(Reward):
    """An example reward that uses changes in the "runtime" observation value
    to compute incremental reward.
    """

    def __init__(self):
        super().__init__(
            id="runtime",
            observation_spaces=["runtime"],
            default_value=0,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        self.baseline_runtime = 0

    def reset(self, benchmark: str, observation_view):
        del benchmark  # unused
        self.baseline_runtime = observation_view["runtime"]

    def update(self, action, observations, observation_view):
        del action  # unused
        del observation_view  # unused
        return float(self.baseline_runtime - observations[0]) / self.baseline_runtime


class SizeReward(Reward):
    """An example reward that uses changes in the "size" observation value
    to compute incremental reward.
    """

    def __init__(self):
        super().__init__(
            id="size",
            observation_spaces=["size"],
            default_value=0,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        self.baseline_size = 0

    def reset(self, benchmark: str, observation_view):
        del benchmark  # unused
        self.baseline_runtime = observation_view["size"]

    def update(self, action, observations, observation_view):
        del action  # unused
        del observation_view  # unused
        return float(self.baseline_size - observations[0]) / self.baseline_size


class UnrollingDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name="benchmark://unrolling-v0",
            license="MIT",
            description="Unrolling example dataset",
            site_data_base=site_data_path(
                "example_dataset"
            ),  # TODO: what should we set this to? we are not using it
        )

        self._benchmarks = {
            "/offsets1": Benchmark.from_file_contents(
                "benchmark://unrolling-v0/offsets1",
                self.preprocess(BENCHMARKS_PATH / "offsets1.c"),
            ),
            "/conv2d": Benchmark.from_file_contents(
                "benchmark://unrolling-v0/conv2d",
                self.preprocess(BENCHMARKS_PATH / "conv2d.c"),
            ),
        }

    @staticmethod
    def preprocess(src: Path) -> bytes:
        """Front a C source through the compiler frontend."""
        # TODO(github.com/facebookresearch/CompilerGym/issues/325): We can skip
        # this pre-processing, or do it on the service side, once support for
        # multi-file benchmarks lands.
        cmd = [
            str(llvm.clang_path()),
            "-E",
            "-o",
            "-",
            "-I",
            str(NEURO_VECTORIZER_HEADER.parent),
            src,
        ]
        for directory in get_system_includes():
            cmd += ["-isystem", str(directory)]
        return subprocess.check_output(
            cmd,
            timeout=300,
        )

    def benchmark_uris(self) -> Iterable[str]:
        yield from (f"benchmark://unrolling-v0{k}" for k in self._benchmarks.keys())

    def benchmark_from_parsed_uri(self, uri: BenchmarkUri) -> Benchmark:
        if uri.path in self._benchmarks:
            return self._benchmarks[uri.path]
        else:
            raise LookupError("Unknown program name")


# Register the unrolling example service on module import. After importing this module,
# the unrolling-py-v0 environment will be available to gym.make(...).

register(
    id="unrolling-py-v0",
    entry_point="compiler_gym.envs:CompilerEnv",
    kwargs={
        "service": UNROLLING_PY_SERVICE_BINARY,
        "rewards": [RuntimeReward(), SizeReward()],
        "datasets": [UnrollingDataset()],
    },
)
