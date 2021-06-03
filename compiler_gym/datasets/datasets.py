# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
from typing import Dict, Iterable, Optional, Set, TypeVar

import numpy as np

from compiler_gym.datasets.benchmark import Benchmark
from compiler_gym.datasets.dataset import Dataset
from compiler_gym.datasets.uri import BENCHMARK_URI_RE, resolve_uri_protocol

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


class Datasets:
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
    """

    def __init__(
        self,
        datasets: Iterable[Dataset],
    ):
        self._datasets: Dict[str, Dataset] = {d.name: d for d in datasets}
        self._visible_datasets: Set[str] = set(
            name for name, dataset in self._datasets.items() if not dataset.deprecated
        )

    def datasets(self, with_deprecated: bool = False) -> Iterable[Dataset]:
        """Enumerate the datasets.

        Dataset order is consistent across runs.

        :param with_deprecated: If :code:`True`, include datasets that have been
            marked as deprecated.

        :return: An iterable sequence of :meth:`Dataset
            <compiler_gym.datasets.Dataset>` instances.
        """
        datasets = self._datasets.values()
        if not with_deprecated:
            datasets = (d for d in datasets if not d.deprecated)
        yield from sorted(datasets, key=lambda d: (d.sort_order, d.name))

    def __iter__(self) -> Iterable[Dataset]:
        """Iterate over the datasets.

        Dataset order is consistent across runs.

        Equivalent to :meth:`datasets.datasets()
        <compiler_gym.datasets.Dataset.datasets>`, but without the ability to
        iterate over the deprecated datasets.

        If the number of benchmarks in any of the datasets is infinite
        (:code:`len(dataset) == math.inf`), the iterable returned by this method
        will continue indefinitely.

        :return: An iterable sequence of :meth:`Dataset
            <compiler_gym.datasets.Dataset>` instances.
        """
        return self.datasets()

    def dataset(self, dataset: str) -> Dataset:
        """Get a dataset.

        Return the corresponding :meth:`Dataset
        <compiler_gym.datasets.Dataset>`. Name lookup will succeed whether or
        not the dataset is deprecated.

        :param dataset: A dataset name.

        :return: A :meth:`Dataset <compiler_gym.datasets.Dataset>` instance.

        :raises LookupError: If :code:`dataset` is not found.
        """
        dataset_name = resolve_uri_protocol(dataset)

        if dataset_name not in self._datasets:
            raise LookupError(f"Dataset not found: {dataset_name}")

        return self._datasets[dataset_name]

    def __getitem__(self, dataset: str) -> Dataset:
        """Lookup a dataset.

        :param dataset: A dataset name.

        :return: A :meth:`Dataset <compiler_gym.datasets.Dataset>` instance.

        :raises LookupError: If :code:`dataset` is not found.
        """
        return self.dataset(dataset)

    def __setitem__(self, key: str, dataset: Dataset):
        """Add a dataset to the collection.

        :param key: The name of the dataset.
        :param dataset: The dataset to add.
        """
        dataset_name = resolve_uri_protocol(key)

        self._datasets[dataset_name] = dataset
        if not dataset.deprecated:
            self._visible_datasets.add(dataset_name)

    def __delitem__(self, dataset: str):
        """Remove a dataset from the collection.

        This does not affect any underlying storage used by dataset. See
        :meth:`uninstall() <compiler_gym.datasets.Datasets.uninstall>` to clean
        up.

        :param dataset: The name of a dataset.

        :return: :code:`True` if the dataset was removed, :code:`False` if it
            was already removed.
        """
        dataset_name = resolve_uri_protocol(dataset)
        if dataset_name in self._visible_datasets:
            self._visible_datasets.remove(dataset_name)
        del self._datasets[dataset_name]

    def __contains__(self, dataset: str) -> bool:
        """Returns whether the dataset is contained."""
        try:
            self.dataset(dataset)
            return True
        except LookupError:
            return False

    def benchmarks(self, with_deprecated: bool = False) -> Iterable[Benchmark]:
        """Enumerate the (possibly infinite) benchmarks lazily.

        Benchmarks order is consistent across runs. One benchmark from each
        dataset is returned in round robin order until all datasets have been
        fully enumerated. The order of :meth:`benchmarks()
        <compiler_gym.datasets.Datasets.benchmarks>` and :meth:`benchmark_uris()
        <compiler_gym.datasets.Datasets.benchmark_uris>` is the same.

        If the number of benchmarks in any of the datasets is infinite
        (:code:`len(dataset) == math.inf`), the iterable returned by this method
        will continue indefinitely.

        :param with_deprecated: If :code:`True`, include benchmarks from
            datasets that have been marked deprecated.

        :return: An iterable sequence of :class:`Benchmark
            <compiler_gym.datasets.Benchmark>` instances.
        """
        return round_robin_iterables(
            (d.benchmarks() for d in self.datasets(with_deprecated=with_deprecated))
        )

    def benchmark_uris(self, with_deprecated: bool = False) -> Iterable[str]:
        """Enumerate the (possibly infinite) benchmark URIs.

        Benchmark URI order is consistent across runs. URIs from datasets are
        returned in round robin order. The order of :meth:`benchmarks()
        <compiler_gym.datasets.Datasets.benchmarks>` and :meth:`benchmark_uris()
        <compiler_gym.datasets.Datasets.benchmark_uris>` is the same.

        If the number of benchmarks in any of the datasets is infinite
        (:code:`len(dataset) == math.inf`), the iterable returned by this method
        will continue indefinitely.

        :param with_deprecated: If :code:`True`, include benchmarks from
            datasets that have been marked deprecated.

        :return: An iterable sequence of benchmark URI strings.
        """
        return round_robin_iterables(
            (d.benchmark_uris() for d in self.datasets(with_deprecated=with_deprecated))
        )

    def benchmark(self, uri: str) -> Benchmark:
        """Select a benchmark.

        Returns the corresponding :class:`Benchmark
        <compiler_gym.datasets.Benchmark>`, regardless of whether the containing
        dataset is installed or deprecated.

        :param uri: The URI of the benchmark to return.

        :return: A :class:`Benchmark <compiler_gym.datasets.Benchmark>`
            instance.
        """
        uri = resolve_uri_protocol(uri)

        match = BENCHMARK_URI_RE.match(uri)
        if not match:
            raise ValueError(f"Invalid benchmark URI: '{uri}'")

        dataset_name = match.group("dataset")
        dataset = self._datasets[dataset_name]

        return dataset.benchmark(uri)

    def random_benchmark(
        self, random_state: Optional[np.random.Generator] = None
    ) -> Benchmark:
        """Select a benchmark randomly.

        First, a dataset is selected uniformly randomly using
        :code:`random_state.choice(list(datasets))`. The
        :meth:`random_benchmark()
        <compiler_gym.datasets.Dataset.random_benchmark>` method of that dataset
        is then called to select a benchmark.

        Note that the distribution of benchmarks selected by this method is not
        biased by the size of each dataset, since datasets are selected
        uniformly. This means that datasets with a small number of benchmarks
        will be overrepresented compared to datasets with many benchmarks. To
        correct for this bias, use the number of benchmarks in each dataset as
        a weight for the random selection:

            >>> rng = np.random.default_rng()
            >>> finite_datasets = [d for d in env.datasets if len(d) != math.inf]
            >>> dataset = rng.choice(
                finite_datasets,
                p=[len(d) for d in finite_datasets]
            )
            >>> dataset.random_benchmark(random_state=rng)

        :param random_state: A random number generator. If not provided, a
            default :code:`np.random.default_rng()` is used.

        :return: A :class:`Benchmark <compiler_gym.datasets.Benchmark>`
            instance.
        """
        random_state = random_state or np.random.default_rng()
        dataset = random_state.choice(list(self._visible_datasets))
        return self[dataset].random_benchmark(random_state=random_state)

    @property
    def size(self) -> int:
        return len(self._visible_datasets)

    def __len__(self) -> int:
        """The number of datasets in the collection."""
        return self.size
