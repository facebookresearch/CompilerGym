# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Iterable, List

from pydantic import BaseModel, Field, validator

from compiler_gym.datasets import Benchmark
from compiler_gym.envs import CompilerEnv
from compiler_gym.wrappers import CycleOverBenchmarks

from .benchmarks import Benchmarks


class Validation(BaseModel):
    """The validation set which is used for periodically evaluating agent
    performance during training.
    """

    # === Start of fields list. ===

    benchmarks: List[Benchmarks] = Field(allow_mutation=False)
    """The benchmarks to evaluate agent performance on. These must be distinct
    from the training and testing sets (this requirement is not enforced by the
    API, you have to do it yourself).
    """

    # === Start of public API. ===

    def benchmarks_iterator(self, env: CompilerEnv) -> Iterable[Benchmark]:
        """Return an iterator over the validation benchmarks."""
        for bm in self.benchmarks:
            yield from bm.benchmarks_iterator(env)

    def benchmark_uris_iterator(self, env: CompilerEnv) -> Iterable[str]:
        """Return an iterator over the training benchmark URIs."""
        for bm in self.benchmarks:
            yield from bm.benchmark_uris_iterator(env)

    def wrap_env(self, env: CompilerEnv) -> CompilerEnv:
        """Wrap an environment for use in the training loop that is configured
        to iterate over the validation benchmarks on each call to
        :code:`reset()`.
        """
        return CycleOverBenchmarks(env=env, benchmarks=self.benchmarks_iterator(env))

    # === Start of implementation details. ===

    @validator("benchmarks", pre=True)
    def validate_benchmarks(cls, value):
        return [Benchmarks(**v) for v in value]

    class Config:
        validate_assignment = True
