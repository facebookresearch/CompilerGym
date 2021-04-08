# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from typing import Dict, Iterable, Optional, Set, Union

import numpy as np

from compiler_gym.datasets.benchmark import Benchmark, resolve_uri_protocol
from compiler_gym.datasets.dataset import BENCHMARK_URI_RE, Dataset


class Datasets(object):
    """A collection of datasets.

    This class serves two goals:

    1. to provide a convenient interface for selecting benchmarks from multiple
       datasets; and

    2. to make it easier to manage multiple datasets by decoupling the selection
       of datasets from their physical installation.

    For the first goal, methods such as :meth:`benchmark()` provide a convenient
    interface for randomly selecting benchmarks across multiple datasets:

        >>> for i in range(100):
        ...     benchmark = env.datasets.benchmark()

    To iterate over a specific dataset, the datasets object can be indexed:

        >>> for i in range(100):
        ...     benchmark = env.datasets["cbench-v1"].benchmark()

    The datasets object can be also be iterated over to list the available
    datasets:

        >>> for dataset in env.datasets:
        ...     print(dataset.name)
        benchmark://cbench-v1
        benchmark://github-v0
        benchmark://npb-v0

    For the second goal, the datasets interface adds an *active* vs *installed*
    datasets abstraction. An active dataset will be included in the
    :meth:`datasets() <compiler_gym.datasets.Datasets.datasets>`,
    :meth:`benchmarks() <compiler_gym.datasets.Datasets.benchmarks>`, and
    :meth:`benchmark_uris() <compiler_gym.datasets.Datasets.benchmark_uris>`
    value iterators. Upon first use, it will be installed, which may require an
    expensive process of downloading remote resources and caching them locally.

    Use :meth:`deactivate() <compiler_gym.datasets.Datasets.deactivate>` to mark
    a dataset as inactive such that it won't be included in set of available
    datasets, but without removing any files that would require a lengthy
    re-installation:

        >>> env.datasets.install("github-v0")
        >>> env.datasets.deactivate("github-v0")
        >>> for dataset in env.datasets:
        ...     print(dataset.name)
        benchmark://cbench-v1
        benchmark://npb-v0
        >>> env.datasets["github-v0"].installed
        True

    The :meth:`activate() <compiler_gym.datasets.Datasets.activate>` method can
    later be used to mark this dataset as ready for use again, without needing
    to reinstall anything:

        >>> env.datasets.activate("github-v0")
        >>> for dataset in env.datasets:
        ...     print(dataset.name)
        benchmark://cbench-v1
        benchmark://github-v0
        benchmark://npb-v0

    The installation of datasets occurs lazily and automatically, however there
    are :meth:`install() <compiler_gym.datasets.Datasets.install>` and
    :meth:`uninstall() <compiler_gym.datasets.Datasets.uninstall>` methods
    should you wish to perform this manually.
    """

    def __init__(
        self,
        datasets: Iterable[Dataset],
        random: Optional[np.random.Generator] = None,
    ):
        # A mapping from dataset name to the dataset instance.
        self._datasets: Dict[str, Dataset] = {d.name: d for d in datasets}
        # The set of activate datasets. All datasets at construction time are
        # initialized as activate.
        self._active_datasets: Set[str] = set(self._datasets.keys())
        self.random = random or np.random.default_rng()

    def seed(self, seed: int) -> None:
        """Set the random state.

        Setting a random state will fix the order that
        :meth:`datasets.benchmark() <compiler_gym.datasets.Datasets.benchmark>`
        returns benchmarks when called without arguments.

        Calling this method recursively calls :meth:`seed()
        <compiler_gym.datasets.Dataset.seed>` on all member datasets.

        :param seed: A number.
        """
        self.random = np.random.default_rng(seed)
        for dataset in self._datasets.values():
            dataset.seed(seed)

    def datasets(
        self, inactive: bool = False, hidden: bool = False
    ) -> Iterable[Dataset]:
        """Enumerate the datasets.

        Dataset order is consistent across runs.

        :param inactive: If :code:`False` (the default), only datasets that are
            active will be returned.

        :param hidden: If :code:`False` (the default), only datasets whose
            :meth:`Dataset.hidden <compiler_gym.datasets.Dataset.hidden>` value
            is :code:`False` are returned.

        :return: An iterable sequence of :meth:`Dataset
            <compiler_gym.datasets.Dataset>` instances.
        """
        datasets = self._datasets.values()
        if not inactive:
            datasets = (d for d in datasets if d.name in self._active_datasets)
        if not hidden:
            datasets = (d for d in datasets if not d.hidden)
        yield from sorted(datasets, key=lambda d: (d.sort_order, d.name))

    def __iter__(self) -> Iterable[Dataset]:
        """Iterate over the active datasets.

        Dataset order is consistent across runs.

        Equivalent to iterating over :meth:`datasets.datasets()
        <compiler_gym.datasets.Dataset.datasets>`, but without the ability to
        iterate over the inactive or hidden datasets.

        :return: An iterable sequence of :meth:`Dataset
            <compiler_gym.datasets.Dataset>` instances.
        """
        yield from self.datasets()

    def dataset(self, name: Optional[str] = None) -> Dataset:
        """Get a dataset.

        If a name is given, return the corresponding :meth:`Dataset
        <compiler_gym.datasets.Dataset>`. Else, return a dataset uniformly
        randomly from the set of available datasets.

        Use :meth:`seed() <compiler_gym.datasets.Dataset.seed>` to force a
        reproducible order for randomly selected datasets.

        Name lookup will succeed whether or not the dataset is active or hidden.

        :param name: A dataset name, or :code:`None` to select a dataset
            randomly.

        :return: A :meth:`Dataset <compiler_gym.datasets.Dataset>` instance.

        :raises LookupError: If :code:`name` is not found.
        """
        if not self._active_datasets:
            raise ValueError("No active datasets available")

        if name is None:
            return self._datasets[self.random.choice(list(self._active_datasets))]

        name = resolve_uri_protocol(name)
        if name not in self._datasets:
            raise LookupError(f"Dataset not found: {name}")
        return self._datasets[name]

    def __getitem__(self, name: str) -> Dataset:
        """Lookup a dataset by name.

        :param name: A dataset name.

        :return: A :meth:`Dataset <compiler_gym.datasets.Dataset>` instance.

        :raises LookupError: If :code:`name` is not found.
        """
        return self.dataset(name)

    def add(self, dataset: Dataset) -> bool:
        """Add a dataset to the datasets collection.

        Once added, a dataset is active. Use :meth:`deactivate()
        <compiler_gym.datasets.Datasets.deactivate>` if you wish for the newly
        added dataset to be inactive.

        Replaces any existing dataset with the same name. A warning is raised if
        a dataset with the same name is replaced.

        :param dataset: A :meth:`Dataset <compiler_gym.datasets.Dataset>`
            instance.

        :return: :code:`True` if the dataset was added, :code:`False` if it was
            already added.

        :raises TypeError: If the given dataset is of incorrect type.
        """
        # Use an attribute check rather than isinstance() to allow mocking the
        # Dataset class in testing.
        if not hasattr(dataset, "name"):
            raise TypeError(
                f"Expected Dataset instance, received: {type(dataset).__name__}"
            )

        added = True
        if dataset.name in self._datasets:
            added = False
            warnings.warn(f"Replacing existing dataset: {dataset.name}")
        self._datasets[dataset.name] = dataset
        self._active_datasets.add(dataset.name)
        return added

    def remove(self, dataset: Union[str, Dataset]) -> bool:
        """Remove a dataset from the collection.

        This does not affect any underlying storage used by dataset. See
        :meth:`uninstall() <compiler_gym.datasets.Datasets.uninstall>` to clean
        up.

        :param dataset: A :meth:`Dataset <compiler_gym.datasets.Dataset>`
            instance, or the name of a dataset.

        :return: :code:`True` if the dataset was removed, :code:`False` if it
            was already removed.
        """
        dataset_name: str = (
            dataset.name
            if isinstance(dataset, Dataset)
            else resolve_uri_protocol(dataset)
        )

        if dataset_name in self._datasets:
            if dataset_name in self._active_datasets:
                self._active_datasets.remove(dataset_name)
            del self._datasets[dataset_name]
            return True

        return False

    def activate(self, dataset: Union[str, Dataset]) -> bool:
        """Activate a dataset.

        Once activated, the dataset will be included in the :meth:`datasets()
        <compiler_gym.datasets.Datasets.datasets>`, :meth:`benchmarks()
        <compiler_gym.datasets.Datasets.benchmarks>`, and
        :meth:`benchmark_uris() <compiler_gym.datasets.Datasets.benchmark_uris>`
        value iterators.

        .. note::

            Activating a dataset does not affect any underlying
            filesystem storage. For that, see :meth:`install()
            <compiler_gym.datasets.Datasets.install>`.

        :param dataset: A :meth:`Dataset <compiler_gym.datasets.Dataset>`
            instance, or the name of a dataset.

        :return: :code:`True` if the dataset was activated, :code:`False` if it
            was already activate.
        """
        dataset_name: str = (
            dataset.name
            if isinstance(dataset, Dataset)
            else resolve_uri_protocol(dataset)
        )
        activated = dataset_name not in self._active_datasets
        self._active_datasets.add(dataset_name)
        return activated

    def is_active(self, dataset: Union[str, Dataset]) -> bool:
        """Returns whether a dataset is active.

        :param dataset: A :meth:`Dataset <compiler_gym.datasets.Dataset>`
            instance, or the name of a dataset.

        :return: :code:`True` if the dataset is activate, else :code:`False`.
        """
        dataset_name: str = (
            dataset.name
            if isinstance(dataset, Dataset)
            else resolve_uri_protocol(dataset)
        )
        return dataset_name in self._active_datasets

    def deactivate(self, dataset: Union[str, Dataset]) -> bool:
        """Deactivate a dataset.

        After deactivating a dataset, it will not be included in the
        :meth:`datasets() <compiler_gym.datasets.Datasets.datasets>`,
        :meth:`benchmarks() <compiler_gym.datasets.Datasets.benchmarks>`, and
        :meth:`benchmark_uris() <compiler_gym.datasets.Datasets.benchmark_uris>`
        value iterators.

        .. note::

            Deactivating a dataset does not affect any underlying
            filesystem storage. Use see :meth:`uninstall()
            <compiler_gym.datasets.Datasets.uninstall>` to remove the files underlying a dataset.

        :param dataset: A :meth:`Dataset <compiler_gym.datasets.Dataset>`
            instance, or the name of a dataset.

        :return: :code:`True` if the dataset was deactivated, :code:`False` if
            it was already inactive.
        """
        dataset_name: str = (
            dataset.name
            if isinstance(dataset, Dataset)
            else resolve_uri_protocol(dataset)
        )
        deactivated = dataset_name in self._active_datasets
        if deactivated:
            self._active_datasets.remove(dataset_name)
        return deactivated

    def install(self, dataset: Union[str, Dataset]) -> bool:
        """Install a dataset.

        Run the local installation procedure for the given dataset. This will
        perform one-time setup of the dataset such as downloading remote
        resources. See :meth:`Dataset.install()
        <compiler_gym.datasets.Dataset.install>`.

        Installing a dataset does not make it inactive if it is not already.

        :param dataset: A :meth:`Dataset <compiler_gym.datasets.Dataset>`
            instance, or the name of a dataset.

        :return: :code:`True` if the dataset was installed, :code:`False` if
            it was already installed.
        """
        dataset: Dataset = (
            dataset if isinstance(dataset, Dataset) else self._datasets[dataset]
        )
        installed = not dataset.installed
        dataset.install()
        return installed

    def uninstall(self, dataset: Union[str, Dataset]) -> bool:
        """Uninstall a dataset.

        Run the local uninstall procedure for the given dataset. This will
        perform one-time clean up of the dataset such as removing locally cached
        files. See :meth:`Dataset.uninstall()
        <compiler_gym.datasets.Dataset.uninstall>`.

        Uninstalling a dataset does not make it inactive, and does not remove it
        from this datasets collection.

        :param dataset: A :meth:`Dataset <compiler_gym.datasets.Dataset>`
            instance, or the name of a dataset.

        :return: :code:`True` if the dataset was uninstalled, :code:`False` if
            it was already uninstalled.
        """
        dataset: Dataset = (
            dataset if isinstance(dataset, Dataset) else self._datasets[dataset]
        )
        uninstalled = dataset.installed
        dataset.uninstall()
        return uninstalled

    def require(self, dataset: Union[str, Dataset]) -> bool:
        """Require that a dataset is available for immediate use.

        This performs any one-time setup of the dataset so that it can be used.
        It is equivalent to:

            >>> env.datasets.add(dataset)
            >>> env.datasets.activate(dataset)
            >>> env.datasets.install(dataset)

        :param dataset: A :meth:`Dataset <compiler_gym.datasets.Dataset>`
            instance, or the name of a dataset.

        :return: :code:`True` if the dataset was made available, :code:`False`
            if it was already available.
        """
        added = False
        # Use an attribute check rather than isinstance() to allow mocking the
        # Dataset class in testing.
        if hasattr(dataset, "name"):
            added = self.add(dataset)
            dataset_name = dataset.name
        else:
            dataset_name = resolve_uri_protocol(dataset)
        activated = self.activate(dataset_name)
        installed = self.install(dataset_name)
        return added or activated or installed

    def benchmarks(self) -> Iterable[Benchmark]:
        """Enumerate the (possibly infinite) benchmarks lazily.

        Benchmarks order is consistent across runs. Benchmarks from each dataset
        are returned in order. The order of :meth:`benchmarks()
        <compiler_gym.datasets.Datasets.benchmarks>` and :meth:`benchmark_uris()
        <compiler_gym.datasets.Datasets.benchmark_uris>` is the same.

        :return: An iterable sequence of :class:`Benchmark
            <compiler_gym.datasets.Benchmark>` instances.
        """
        for dataset in self.datasets():
            yield from dataset.benchmarks()

    def benchmark_uris(self) -> Iterable[str]:
        """Enumerate the (possibly infinite) benchmark URIs.

        Benchmark URI order is consistent across runs. URIs from each dataset
        are returned in order. The order of :meth:`benchmarks()
        <compiler_gym.datasets.Datasets.benchmarks>` and :meth:`benchmark_uris()
        <compiler_gym.datasets.Datasets.benchmark_uris>` is the same.

        :return: An iterable sequence of benchmark URI strings.
        """
        for dataset in self.datasets():
            yield from dataset.benchmark_uris()

    def benchmark(self, uri: Optional[str] = None) -> Benchmark:
        """Select a benchmark.

        If a benchmark URI is given, the corresponding :class:`Benchamrk
        <compiler_gym.datasets.Benchmark>` is returned, regardless of whether
        the containing dataset is active, hidden, or installed.

        If no URI is given, a benchmark is selected randomly. Fist, a dataset is
        selected uniformly randomly from the set of available datasets. Then a
        benchmark is selected randomly from the chosen dataset.

        Calling :code:`benchmark()` will yield benchmarks from all available
        datasets with equal probability, regardless of how many benchmarks are
        in each dataset. Given a pool of available datasets of differing sizes,
        smaller datasets will be overrepresented and large datasets will be
        underrepresented.

        Use :meth:`seed() <compiler_gym.datasets.Dataset.seed>` to force a
        reproducible order for randomly selected benchmarks.

        :param uri: The URI of the benchmark to return. If :code:`None`, select
            a benchmark randomly using :code:`self.random`.

        :return: A :class:`Benchamrk <compiler_gym.datasets.Benchmark>`
            instance.
        """
        if not self._active_datasets:
            raise ValueError("No active datasets available")

        if uri is None:
            return self.dataset().benchmark()

        uri = resolve_uri_protocol(uri)

        match = BENCHMARK_URI_RE.match(uri)
        if not match:
            raise ValueError(f"Invalid benchmark URI: '{uri}'")

        dataset_name = match.group("dataset")
        if dataset_name not in self._datasets:
            raise LookupError(f"Dataset not found: '{dataset_name}'")

        return self._datasets[dataset_name].benchmark(uri)
