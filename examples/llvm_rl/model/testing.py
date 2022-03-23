# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from itertools import islice
from typing import Iterable, List

import numpy as np
from pydantic import BaseModel, Field, validator

from compiler_gym.datasets import Benchmark
from compiler_gym.envs import CompilerEnv

from .benchmarks import Benchmarks

logger = logging.getLogger(__name__)


class Testing(BaseModel):
    """The testing regime."""

    __test__ = False  # Prevent pytest from thinking that this class is a test.

    # === Start of fields list. ===

    timeout_hours: float = Field(allow_mutation=False, gt=0)
    """The timeout for test jobs, in hours."""

    benchmarks: List[Benchmarks] = Field(allow_mutation=False)
    """The set of benchmarks to test on."""

    runs_per_benchmark: int = Field(default=1, ge=1, allow_mutation=False)
    """The number of inference episodes to run on each benchmark. If the
    environment and policy are deterministic then running multiple episodes per
    benchmark is only useful for producing accurate aggregate measurements of
    inference walltime.
    """

    # === Start of public API. ===

    def benchmarks_iterator(self, env: CompilerEnv) -> Iterable[Benchmark]:
        """Return an iterator over the test benchmarks."""
        for _ in range(self.runs_per_benchmark):
            for bm in self.benchmarks:
                yield from bm.benchmarks_iterator(env)

    def benchmark_uris_iterator(self, env: CompilerEnv) -> Iterable[str]:
        """Return an iterator over the test benchmark URIs."""
        for _ in range(self.runs_per_benchmark):
            for bm in self.benchmarks:
                yield from bm.benchmark_uris_iterator(env)

    # === Start of implementation details. ===

    @validator("benchmarks", pre=True)
    def validate_benchmarks(cls, value):
        return [Benchmarks(**v) for v in value]

    class Config:
        validate_assignment = True


def get_testing_benchmarks(
    env: CompilerEnv, max_benchmarks: int = 50, seed: int = 0
) -> List[str]:
    rng = np.random.default_rng(seed=seed)
    for dataset in env.datasets:
        if dataset.name == "generator://csmith-v0":
            yield from islice(dataset.benchmarks(), 50)
        elif not dataset.size or dataset.size > max_benchmarks:
            logger.info(
                "Selecting random %d benchmarks from dataset %s of size %d",
                max_benchmarks,
                dataset,
                dataset.size,
            )
            for _ in range(max_benchmarks):
                yield dataset.random_benchmark(rng)
        else:
            logger.info(
                "Selecting all %d benchmarks from dataset %s", dataset.size, dataset
            )
            yield from dataset.benchmarks()
