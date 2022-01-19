# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from itertools import cycle
from typing import Callable, Iterable, Optional, Union

import numpy as np

from compiler_gym.datasets import Benchmark
from compiler_gym.envs import CompilerEnv
from compiler_gym.util.parallelization import thread_safe_tee
from compiler_gym.wrappers.core import CompilerEnvWrapper

BenchmarkLike = Union[str, Benchmark]


class IterateOverBenchmarks(CompilerEnvWrapper):
    """Iterate over a (possibly infinite) sequence of benchmarks on each call to
    reset(). Will raise :code:`StopIteration` on :meth:`reset()
    <compiler_gym.envs.CompilerEnv.reset>` once the iterator is exhausted. Use
    :class:`CycleOverBenchmarks` or :class:`RandomOrderBenchmarks` for wrappers
    which will loop over the benchmarks.
    """

    def __init__(
        self,
        env: CompilerEnv,
        benchmarks: Iterable[BenchmarkLike],
        fork_shares_iterator: bool = False,
    ):
        """Constructor.

        :param env: The environment to wrap.

        :param benchmarks: An iterable sequence of benchmarks.

        :param fork_shares_iterator: If :code:`True`, the :code:`benchmarks`
            iterator will bet shared by a forked environment created by
            :meth:`env.fork() <compiler_gym.envs.CompilerEnv.fork>`. This means
            that calling :meth:`env.reset()
            <compiler_gym.envs.CompilerEnv.reset>` with one environment will
            advance the iterator in the other. If :code:`False`, forked
            environments will use :code:`itertools.tee()` to create a copy of
            the iterator so that each iterator may advance independently.
            However, this requires shared buffers between the environments which
            can lead to memory overheads if :meth:`env.reset()
            <compiler_gym.envs.CompilerEnv.reset>` is called many times more in
            one environment than the other.
        """
        super().__init__(env)
        self.benchmarks = iter(benchmarks)
        self.fork_shares_iterator = fork_shares_iterator

    def reset(self, benchmark: Optional[BenchmarkLike] = None, **kwargs):
        if benchmark is not None:
            raise TypeError("Benchmark passed to IterateOverBenchmarks.reset()")
        benchmark: BenchmarkLike = next(self.benchmarks)
        return self.env.reset(benchmark=benchmark)

    def fork(self) -> "IterateOverBenchmarks":
        if self.fork_shares_iterator:
            other_benchmarks_iterator = self.benchmarks
        else:
            self.benchmarks, other_benchmarks_iterator = thread_safe_tee(
                self.benchmarks
            )
        return IterateOverBenchmarks(
            env=self.env.fork(),
            benchmarks=other_benchmarks_iterator,
            fork_shares_iterator=self.fork_shares_iterator,
        )


class CycleOverBenchmarks(IterateOverBenchmarks):
    """Cycle through a list of benchmarks on each call to :meth:`reset()
    <compiler_gym.envs.CompilerEnv.reset>`. Same as
    :class:`IterateOverBenchmarks` except the list of benchmarks repeats once
    exhausted.
    """

    def __init__(
        self,
        env: CompilerEnv,
        benchmarks: Iterable[BenchmarkLike],
        fork_shares_iterator: bool = False,
    ):
        """Constructor.

        :param env: The environment to wrap.

        :param benchmarks: An iterable sequence of benchmarks.

        :param fork_shares_iterator: If :code:`True`, the :code:`benchmarks`
            iterator will be shared by a forked environment created by
            :meth:`env.fork() <compiler_gym.envs.CompilerEnv.fork>`. This means
            that calling :meth:`env.reset()
            <compiler_gym.envs.CompilerEnv.reset>` with one environment will
            advance the iterator in the other. If :code:`False`, forked
            environments will use :code:`itertools.tee()` to create a copy of
            the iterator so that each iterator may advance independently.
            However, this requires shared buffers between the environments which
            can lead to memory overheads if :meth:`env.reset()
            <compiler_gym.envs.CompilerEnv.reset>` is called many times more in
            one environment than the other.
        """
        super().__init__(
            env, benchmarks=cycle(benchmarks), fork_shares_iterator=fork_shares_iterator
        )


class CycleOverBenchmarksIterator(CompilerEnvWrapper):
    """Same as :class:`CycleOverBenchmarks
    <compiler_gym.wrappers.CycleOverBenchmarks>` except that the user generates
    the iterator.
    """

    def __init__(
        self,
        env: CompilerEnv,
        make_benchmark_iterator: Callable[[], Iterable[BenchmarkLike]],
    ):
        """Constructor.

        :param env: The environment to wrap.

        :param make_benchmark_iterator: A callback that returns an iterator over
            a sequence of benchmarks. Once the iterator is exhausted, this
            callback is called to produce a new iterator.
        """
        super().__init__(env)
        self.make_benchmark_iterator = make_benchmark_iterator
        self.benchmarks = iter(self.make_benchmark_iterator())

    def reset(self, benchmark: Optional[BenchmarkLike] = None, **kwargs):
        if benchmark is not None:
            raise TypeError("Benchmark passed toIterateOverBenchmarks.reset()")
        try:
            benchmark: BenchmarkLike = next(self.benchmarks)
        except StopIteration:
            self.benchmarks = iter(self.make_benchmark_iterator())
            benchmark: BenchmarkLike = next(self.benchmarks)

        return self.env.reset(benchmark=benchmark)

    def fork(self) -> "CycleOverBenchmarksIterator":
        return CycleOverBenchmarksIterator(
            env=self.env.fork(),
            make_benchmark_iterator=self.make_benchmark_iterator,
        )


class RandomOrderBenchmarks(IterateOverBenchmarks):
    """Select randomly from a list of benchmarks on each call to :meth:`reset()
    <compiler_gym.envs.CompilerEnv.reset>`.

    .. note::

        Uniform random selection is provided by evaluating the input benchmarks
        iterator into a list and sampling randomly from the list. For very large
        and infinite iterables of benchmarks you must use the
        :class:`IterateOverBenchmarks
        <compiler_gym.wrappers.IterateOverBenchmarks>` wrapper with your own
        random sampling iterator.
    """

    def __init__(
        self,
        env: CompilerEnv,
        benchmarks: Iterable[BenchmarkLike],
        rng: Optional[np.random.Generator] = None,
    ):
        """Constructor.

        :param env: The environment to wrap.

        :param benchmarks: An iterable sequence of benchmarks. The entirety of
            this input iterator is evaluated during construction.

        :param rng: A random number generator to use for random benchmark
            selection.
        """
        self._all_benchmarks = list(benchmarks)
        rng = rng or np.random.default_rng()
        super().__init__(
            env,
            benchmarks=(rng.choice(self._all_benchmarks) for _ in iter(int, 1)),
            fork_shares_iterator=True,
        )

    def fork(self) -> "IterateOverBenchmarks":
        """Fork the random order benchmark wrapper.

        Note that RNG state is not copied to forked environments.
        """
        return IterateOverBenchmarks(
            env=self.env.fork(), benchmarks=self._all_benchmarks
        )
