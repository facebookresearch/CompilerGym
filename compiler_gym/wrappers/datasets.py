# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from itertools import cycle
from typing import Iterable, Optional, Union

import numpy as np

from compiler_gym.datasets import Benchmark
from compiler_gym.envs import CompilerEnv
from compiler_gym.wrappers.core import CompilerEnvWrapper

BenchmarkArg = Union[str, Benchmark]


class IterateOverBenchmarks(CompilerEnvWrapper):
    """Iterate over a (possibly finite) sequence of benchmarks on each call to
    reset(). Will raise :code:`StopIteration` on :meth:`reset()
    <compiler_gym.envs.CompilerEnv.reset>` once the iterator is exhausted. Use
    :class:`CycleOverBenchmarks` or :class:`RandomOrderBenchmarks` for wrappers
    which will loop over the benchmarks.
    """

    def __init__(self, env: CompilerEnv, benchmarks: Iterable[BenchmarkArg]):
        """Constructor.

        :param env: The environment to wrap.

        :param benchmarks: An iterable sequence of benchmarks.
        """
        super().__init__(env)
        self.benchmarks = iter(benchmarks)

    def reset(self, benchmark: Optional[BenchmarkArg] = None, **kwargs):
        if benchmark is not None:
            raise TypeError("Benchmark passed toIterateOverBenchmarks.reset()")
        benchmark: BenchmarkArg = next(self.benchmarks)
        return self.env.reset(benchmark=benchmark)


class CycleOverBenchmarks(IterateOverBenchmarks):
    """Cycle through a list of benchmarks on each call to :meth:`reset()
    <compiler_gym.envs.CompilerEnv.reset>`. Same as
    :class:`IterateOverBenchmarks` except the list of benchmarks repeats once
    exhausted.
    """

    def __init__(
        self,
        env: CompilerEnv,
        benchmarks: Iterable[BenchmarkArg],
    ):
        """Constructor.

        :param env: The environment to wrap.

        :param benchmarks: An iterable sequence of benchmarks.
        """
        super().__init__(env, benchmarks=cycle(benchmarks))


class RandomOrderBenchmarks(IterateOverBenchmarks):
    """Select randomly from a list of benchmarks on each call to :meth:`reset()
    <compiler_gym.envs.CompilerEnv.reset>`.
    """

    def __init__(
        self,
        env: CompilerEnv,
        benchmarks: Iterable[BenchmarkArg],
        rng: Optional[np.random.Generator] = None,
    ):
        """Constructor.

        :param env: The environment to wrap.

        :param benchmarks: An iterable sequence of benchmarks.

        :param rng: A random number generator to use for random benchmark
            selection.
        """
        benchmarks = list(benchmarks)
        rng = rng or np.random.default_rng()
        super().__init__(env, benchmarks=(rng.choice(benchmarks) for _ in iter(int, 1)))
