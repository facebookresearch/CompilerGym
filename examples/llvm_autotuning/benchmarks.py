# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from itertools import islice
from typing import Iterable, List, Union

from pydantic import BaseModel, Field, root_validator, validator

from compiler_gym.datasets import Benchmark, BenchmarkUri
from compiler_gym.envs.llvm import LlvmEnv


class BenchmarksEntry(BaseModel):
    """This class represents a single entry in a Benchmarks class."""

    # === Start of fields list. ===

    dataset: str = Field(default=None, allow_mutation=False)
    """The name of a dataset to iterate over. If set, benchmarks are produced
    by iterating over this dataset in order. If not set, the :code:`uris` list
    must be provided.
    """

    uris: List[str] = Field(default=[], allow_mutation=False)
    """A list of URIs to iterate over."""

    max_benchmarks: int = Field(default=0, ge=0, allow_mutation=False)
    """The maximum number of benchmarks to yield from the given dataset or URIs
    list.
    """

    benchmarks_start_at: int = Field(default=0, ge=0, allow_mutation=False)
    """An offset into the dataset or URIs list to start iterating from.

    Note that using very large offsets will slow things down as the
    implementation still has to iterate over the excluded benchmarks.
    """

    # === Start of public API. ===

    def benchmarks_iterator(self, env: LlvmEnv) -> Iterable[Benchmark]:
        """Return an iterator over the benchmarks."""
        return self._benchmark_iterator(env)

    def benchmark_uris_iterator(self, env: LlvmEnv) -> Iterable[str]:
        """Return an iterator over the URIs of the benchmarks."""
        return self._benchmark_iterator(env, uris=True)

    # === Start of implementation details. ===

    @root_validator
    def check_that_either_dataset_or_uris_is_set(cls, values):
        assert values.get("dataset") or values.get(
            "uris"
        ), "Neither dataset or uris given"
        return values

    @validator("uris", pre=True)
    def validate_uris(cls, value, *, values, **kwargs):
        del kwargs
        del values
        for uri in value:
            uri = BenchmarkUri.from_string(uri)
            assert uri.scheme and uri.dataset, f"Invalid benchmark URI: {uri}"
        return list(value)

    def _benchmark_iterator(
        self, env: LlvmEnv, uris: bool = False
    ) -> Union[Iterable[Benchmark], Iterable[str]]:
        return (
            self._uris_iterator(env, uris)
            if self.uris
            else self._dataset_iterator(env, uris)
        )

    def _uris_iterator(
        self, env: LlvmEnv, uris: bool = False
    ) -> Union[Iterable[Benchmark], Iterable[str]]:
        """Iterate from a URIs list."""
        start = self.benchmarks_start_at
        n = len(self.uris)
        if self.max_benchmarks:
            n = min(len(self.uris), n)

        if uris:
            # Shortcut in case we already have a list of URIs that we can slice
            # rather than iterating over.
            return iter(self.uris[start:n])

        return islice((env.datasets.benchmark(u) for u in self.uris), start, start + n)

    def _dataset_iterator(
        self, env: LlvmEnv, uris: bool = False
    ) -> Union[Iterable[Benchmark], Iterable[str]]:
        """Iterate from a dataset name."""
        dataset = env.datasets[self.dataset]
        dataset.install()
        n = dataset.size or self.max_benchmarks  # dataset.size == 0 for inf
        if self.max_benchmarks:
            n = min(self.max_benchmarks, n)
        start = self.benchmarks_start_at
        iterator = dataset.benchmark_uris if uris else dataset.benchmarks
        return islice(iterator(), start, start + n)

    class Config:
        validate_assignment = True


class Benchmarks(BaseModel):
    """Represents a set of benchmarks to use for training/validation/testing.

    There are two ways of describing benchmarks, either as a list of benchmark
    URIs:

        benchmarks:
            uris:
                - benchmark://cbench-v1/adpcm
                - benchmark://cbench-v1/ghostscript

    Or as a dataset to iterate over:

        benchmarks:
            dataset: benchmark://cbench-v1
            max_benchmarks: 20
    """

    benchmarks: List[BenchmarksEntry]

    # === Start of public API. ===

    def benchmarks_iterator(self, env: LlvmEnv) -> Iterable[Benchmark]:
        """Return an iterator over the benchmarks."""
        for bm in self.benchmarks:
            yield from bm.benchmarks_iterator(env)

    def benchmark_uris_iterator(self, env: LlvmEnv) -> Iterable[str]:
        """Return an iterator over the URIs of the benchmarks."""
        for bm in self.benchmarks:
            yield from bm.benchmark_uris_iterator(env)

    # === Start of implementation details. ===

    @validator("benchmarks", pre=True)
    def validate_benchmarks(cls, value) -> List[BenchmarksEntry]:
        return [BenchmarksEntry(**v) for v in value]
