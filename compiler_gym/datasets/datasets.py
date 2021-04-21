# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
from typing import Dict, Iterable, Optional, Set, TypeVar, Union

import numpy as np

from compiler_gym.datasets.benchmark import (
    BENCHMARK_URI_RE,
    Benchmark,
    resolve_uri_protocol,
)
from compiler_gym.datasets.dataset import Dataset

T = TypeVar("T")


def round_robin_iterables(iters: Iterable[Iterable[T]]) -> Iterable[T]:
    """Yield from the given iterators in round robin order."""
    # Use a queue of iterators to iterate over. Repeatedly pop an iterator from
    # the queue, yield the next value from it, then put it at the back of the
    # queue. The iterator is discarded once exhausted.
    iters = deque(iters)
    while len(iters) > 1:
        it = iters.popleft()
        try:
            yield next(it)
            iters.append(it)
        except StopIteration:
            pass
    # Once we have only a single iterator left, return it directly rather
    # continuing with the round robin.
    if len(iters) == 1:
        yield from iters.popleft()


class Datasets(object):
    """A collection of datasets.

    This class provides a dictionary-like interface for indexing and iterating
    over multiple :class:`Dataset <compiler_gym.datasets.Dataset>` objects.
    Select a dataset by URI using:

        >>> env.datasets["benchmark://cbench-v1"]

    Check whether a dataset exists using:

        >>> "benchmark://cbench-v1" in env.datasets
        True

    Or iterate over the datasets using:

        >>> for dataset in env.datasets:
        ...     print(dataset.name)
        benchmark://cbench-v1
        benchmark://github-v0
        benchmark://npb-v0

    To select a benchmark from the datasets, use :meth:`benchmark()`:

        >>> env.datasets.benchmark("benchmark://a-v0/a")

    Use the :meth:`benchmarks()` method to iterate over every benchmark in the
    datasets in a stable round robin order:

        >>> for benchmark in env.datasets.benchmarks():
        ...     print(benchmark)
        benchmark://cbench-v1/1
        benchmark://github-v0/1
        benchmark://npb-v0/1
        benchmark://cbench-v1/2
        ...

    If you want to exclude a dataset, delete it:

        >>> del env.datasets["benchmark://b-v0"]

    To iterate over the benchmarks in a random order, use :meth:`benchmark()`
    and omit the URI:

        >>> for i in range(100):
        ...     benchmark = env.datasets.benchmark()

    This uses uniform random selection to sample across datasets. For finite
    datasets, you could weight the sample by the size of each dataset:

        >>> weights = [len(d) for d in env.datasets]
        >>> np.random.choice(list(env.datasets), p=weights).benchmark()
    """

    def __init__(
        self,
        datasets: Iterable[Dataset],
        random: Optional[np.random.Generator] = None,
    ):
        self._datasets: Dict[str, Dataset] = {d.name: d for d in datasets}
        self._visible_datasets: Set[str] = set(
            name for name, dataset in self._datasets.items() if not dataset.hidden
        )
        self.random = random or np.random.default_rng()

    def seed(self, seed: Optional[int] = None) -> None:
        """Set the random state.

        Setting a random state will fix the order that
        :meth:`datasets.benchmark() <compiler_gym.datasets.Datasets.benchmark>`
        returns benchmarks when called without arguments.

        Calling this method recursively calls :meth:`seed()
        <compiler_gym.datasets.Dataset.seed>` on all member datasets.

        :param seed: An optional seed value.
        """
        self.random = np.random.default_rng(seed)
        for dataset in self._datasets.values():
            dataset.seed(seed)

    def datasets(self, hidden: bool = False) -> Iterable[Dataset]:
        """Enumerate the datasets.

        Dataset order is consistent across runs.

        :param hidden: If :code:`False` (the default), only datasets whose
            :meth:`Dataset.hidden <compiler_gym.datasets.Dataset.hidden>` value
            is :code:`False` are returned.

        :return: An iterable sequence of :meth:`Dataset
            <compiler_gym.datasets.Dataset>` instances.
        """
        datasets = self._datasets.values()
        if not hidden:
            datasets = (d for d in datasets if not d.hidden)
        yield from sorted(datasets, key=lambda d: (d.sort_order, d.name))

    def __iter__(self) -> Iterable[Dataset]:
        """Iterate over the datasets.

        Dataset order is consistent across runs.

        Equivalent to :meth:`datasets.datasets()
        <compiler_gym.datasets.Dataset.datasets>`, but without the ability to
        iterate over the hidden datasets.

        :return: An iterable sequence of :meth:`Dataset
            <compiler_gym.datasets.Dataset>` instances.
        """
        return self.datasets()

    def dataset(self, dataset: Optional[Union[str, Dataset]] = None) -> Dataset:
        """Get a dataset.

        If a name is given, return the corresponding :meth:`Dataset
        <compiler_gym.datasets.Dataset>`. Else, return a dataset uniformly
        randomly from the set of available datasets.

        Use :meth:`seed() <compiler_gym.datasets.Dataset.seed>` to force a
        reproducible order for randomly selected datasets.

        Name lookup will succeed whether or not the dataset is active or hidden.

        :param dataset: A dataset name, a :class:`Dataset` instance, or
            :code:`None` to select a dataset randomly.

        :return: A :meth:`Dataset <compiler_gym.datasets.Dataset>` instance.

        :raises LookupError: If :code:`dataset` is not found.
        """
        if dataset is None:
            if not self._visible_datasets:
                raise ValueError("No datasets")

            return self._datasets[self.random.choice(list(self._visible_datasets))]

        if isinstance(dataset, Dataset):
            dataset_name = dataset.name
        else:
            dataset_name = resolve_uri_protocol(dataset)

        if dataset_name not in self._datasets:
            raise LookupError(f"Dataset not found: {dataset_name}")

        return self._datasets[dataset_name]

    def __getitem__(self, dataset: Union[str, Dataset]) -> Dataset:
        """Lookup a dataset.

        :param dataset: A dataset name, a :class:`Dataset` instance, or
            :code:`None` to select a dataset randomly.

        :return: A :meth:`Dataset <compiler_gym.datasets.Dataset>` instance.

        :raises LookupError: If :code:`dataset` is not found.
        """
        return self.dataset(dataset)

    def __setitem__(self, key: str, dataset: Dataset):
        self._datasets[key] = dataset
        if not dataset.hidden:
            self._visible_datasets.add(dataset.name)

    def __delitem__(self, dataset: Union[str, Dataset]):
        """Remove a dataset from the collection.

        This does not affect any underlying storage used by dataset. See
        :meth:`uninstall() <compiler_gym.datasets.Datasets.uninstall>` to clean
        up.

        :param dataset: A :meth:`Dataset <compiler_gym.datasets.Dataset>`
            instance, or the name of a dataset.

        :return: :code:`True` if the dataset was removed, :code:`False` if it
            was already removed.
        """
        dataset_name: str = self.dataset(dataset).name
        if dataset_name in self._visible_datasets:
            self._visible_datasets.remove(dataset_name)
        del self._datasets[dataset_name]

    def __contains__(self, dataset: Union[str, Dataset]) -> bool:
        """Returns whether the dataset is contained."""
        try:
            self.dataset(dataset)
            return True
        except LookupError:
            return False

    def benchmarks(self) -> Iterable[Benchmark]:
        """Enumerate the (possibly infinite) benchmarks lazily.

        Benchmarks order is consistent across runs. One benchmark from each
        dataset is returned in round robin order until all datasets have been
        fully enumerated. The order of :meth:`benchmarks()
        <compiler_gym.datasets.Datasets.benchmarks>` and :meth:`benchmark_uris()
        <compiler_gym.datasets.Datasets.benchmark_uris>` is the same.

        :return: An iterable sequence of :class:`Benchmark
            <compiler_gym.datasets.Benchmark>` instances.
        """
        return round_robin_iterables((d.benchmarks() for d in self.datasets()))

    def benchmark_uris(self) -> Iterable[str]:
        """Enumerate the (possibly infinite) benchmark URIs.

        Benchmark URI order is consistent across runs. URIs from datasets are
        returned in round robin order. The order of :meth:`benchmarks()
        <compiler_gym.datasets.Datasets.benchmarks>` and :meth:`benchmark_uris()
        <compiler_gym.datasets.Datasets.benchmark_uris>` is the same.

        :return: An iterable sequence of benchmark URI strings.
        """
        return round_robin_iterables((d.benchmark_uris() for d in self.datasets()))

    def benchmark(self, uri: Optional[str] = None) -> Benchmark:
        """Select a benchmark.

        If a benchmark URI is given, the corresponding :class:`Benchmark
        <compiler_gym.datasets.Benchmark>` is returned, regardless of whether
        the containing dataset is active, hidden, or installed.

        If no URI is given, a benchmark is selected randomly. First, a dataset
        is selected uniformly randomly from the set of available datasets. Then
        a benchmark is selected randomly from the chosen dataset.

        Calling :code:`benchmark()` will yield benchmarks from all available
        datasets with equal probability, regardless of how many benchmarks are
        in each dataset. Given a pool of available datasets of differing sizes,
        smaller datasets will be overrepresented and large datasets will be
        underrepresented.

        Use :meth:`seed() <compiler_gym.datasets.Dataset.seed>` to force a
        reproducible order for randomly selected benchmarks.

        :param uri: The URI of the benchmark to return. If :code:`None`, select
            a benchmark randomly using :code:`self.random`.

        :return: A :class:`Benchmark <compiler_gym.datasets.Benchmark>`
            instance.
        """
        if uri is None and not self._visible_datasets:
            raise ValueError("No datasets")
        elif uri is None:
            return self.dataset().benchmark()

        uri = resolve_uri_protocol(uri)

        match = BENCHMARK_URI_RE.match(uri)
        if not match:
            raise ValueError(f"Invalid benchmark URI: '{uri}'")

        dataset_name = match.group("dataset")
        dataset = self._datasets[dataset_name]

        if len(uri) > len(dataset_name) + 1:
            return dataset.benchmark(uri)
        else:
            return dataset.benchmark()

    @property
    def size(self) -> int:
        return len(self._visible_datasets)

    def __len__(self) -> int:
        return self.size
