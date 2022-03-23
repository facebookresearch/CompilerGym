# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Iterable, List

from pydantic import BaseModel, Field, validator

from compiler_gym.datasets import Benchmark
from compiler_gym.envs import CompilerEnv
from compiler_gym.wrappers import (
    CycleOverBenchmarks,
    CycleOverBenchmarksIterator,
    IterateOverBenchmarks,
)

from .benchmarks import Benchmarks
from .validation import Validation


class Training(BaseModel):
    """The training regime."""

    timeout_hours: float = Field(allow_mutation=False, gt=0)
    """The maximum runtime of the training job."""

    episodes: int = Field(ge=1, allow_mutation=False)
    """The number of episodes to train for."""

    benchmarks: List[Benchmarks] = Field(allow_mutation=False)
    """The programs to train over."""

    validation: Validation = Field(allow_mutation=False)
    """The validation set."""

    cycle_over_benchmarks: bool = Field(default=True, allow_mutation=False)
    """If :code:`True`, the benchmark iterator repeats itself once an entire
    epoch has completed. Set this to :code:`False` to disable benchmarks from
    being cached.
    """

    cache_benchmarks: bool = Field(default=False, allow_mutation=False)
    """If :code:`True`, construct the actual benchmark objects during iteration.
    This will make it faster to cycle over the same set of benchmarks multiple
    times, but requires enough resources to hold all of the benchmark objects in
    memory. If :code:`False`, just the benchmark URIs are cached in memory.
    """

    # === Start of public API. ===

    def benchmarks_iterator(self, env: CompilerEnv) -> Iterable[Benchmark]:
        """Return an iterator over the training benchmarks."""
        for bm in self.benchmarks:
            yield from bm.benchmarks_iterator(env)

    def benchmark_uris_iterator(self, env: CompilerEnv) -> Iterable[str]:
        """Return an iterator over the training benchmark URIs."""
        for bm in self.benchmarks:
            yield from bm.benchmark_uris_iterator(env)

    def wrap_env(self, env: CompilerEnv) -> CompilerEnv:
        """Wrap an environment for use in the training loop that is configured
        to iterate over the training benchmarks on each call to :code:`reset()`.
        """
        if self.cycle_over_benchmarks and self.cache_benchmarks:
            wrapper = CycleOverBenchmarks
        elif self.cycle_over_benchmarks:
            return CycleOverBenchmarksIterator(
                env=env,
                make_benchmark_iterator=lambda: self.benchmark_uris_iterator(env),
            )
        else:
            wrapper = IterateOverBenchmarks
        iterator = (
            self.benchmarks_iterator
            if self.cache_benchmarks
            else self.benchmark_uris_iterator
        )
        return wrapper(env=env, benchmarks=iterator(env))

    # === Start of implementation details. ===

    @validator("benchmarks", pre=True)
    def validate_benchmarks(cls, value):
        return [Benchmarks(**v) for v in value]

    class Config:
        validate_assignment = True
