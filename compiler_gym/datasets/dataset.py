# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import numpy as np

# The "deprecated" name is used as a constructor argument to Dataset, so rename
# this import to prevent shadowing.
from deprecated.sphinx import deprecated as mark_deprecated

from compiler_gym.datasets.benchmark import Benchmark
from compiler_gym.datasets.uri import BenchmarkUri

logger = logging.getLogger(__name__)

# NOTE(cummins): This is only required to prevent a name conflict with the now
# deprecated Dataset.logger attribute. This can be removed once the logger
# attribute is removed, scheduled for release 0.2.3.
_logger = logger

_DATASET_VERSION_PATTERN = r"[a-zA-z0-9-_]+-v(?P<version>[0-9]+)"
_DATASET_VERSION_RE = re.compile(_DATASET_VERSION_PATTERN)


class Dataset:
    """A dataset is a collection of benchmarks.

    The Dataset class has methods for installing and managing groups of
    benchmarks, for listing the available benchmark URIs, and for instantiating
    :class:`Benchmark <compiler_gym.datasets.Benchmark>` objects.

    The Dataset class is an abstract base for implementing datasets. At a
    minimum, subclasses must implement the :meth:`benchmark()
    <compiler_gym.datasets.Dataset.benchmark>` and :meth:`benchmark_uris()
    <compiler_gym.datasets.Dataset.benchmark_uris>` methods, and :meth:`size
    <compiler_gym.datasets.Dataset.size>`. Other methods such as
    :meth:`install() <compiler_gym.datasets.Dataset.install>` may be used where
    helpful.
    """

    def __init__(
        self,
        name: str,
        description: str,
        license: str,  # pylint: disable=redefined-builtin
        site_data_base: Optional[Path] = None,
        benchmark_class=Benchmark,
        references: Optional[Dict[str, str]] = None,
        deprecated: Optional[str] = None,
        sort_order: int = 0,
        validatable: str = "No",
    ):
        """Constructor.

        :param name: The name of the dataset, in the format:
            :code:`scheme://name`.

        :param description: A short human-readable description of the dataset.

        :param license: The name of the dataset's license.

        :param site_data_base: An optional directory that can be used by the
            dataset to house the "site data", i.e. persistent files on disk. The
            site data directory is a subdirectory of this :code:`site_data_base`
            path, which can be shared by multiple datasets. If not provided, the
            :attr:`dataset.site_data_path
            <compiler_gym.datasets.Dataset.site_data_path>` attribute will raise
            an error. Use :attr:`dataset.has_site_data
            <compiler_gym.datasets.Dataset.has_site_data>` to check if a site
            data path was set.

        :param benchmark_class: The class to use when instantiating benchmarks.
            It must have the same constructor signature as :class:`Benchmark
            <compiler_gym.datasets.Benchmark>`.

        :param references: A dictionary of useful named URLs for this dataset
            containing extra information, download links, papers, etc.

        :param deprecated: Mark the dataset as deprecated and issue a warning
            when :meth:`install() <compiler_gym.datasets.Dataset.install>`,
            including the given method. Deprecated datasets are excluded from
            the :meth:`datasets() <compiler_gym.datasets.Datasets.dataset>`
            iterator by default.

        :param sort_order: An optional numeric value that should be used to
            order this dataset relative to others. Lowest value sorts first.

        :param validatable: Whether the dataset is validatable. A validatable
            dataset is one where the behavior of the benchmarks can be checked
            by compiling the programs to binaries and executing them. If the
            benchmarks crash, or are found to have different behavior, then
            validation fails. This type of validation is used to check that the
            compiler has not broken the semantics of the program. This value
            takes a string and is used for documentation purposes only.
            Suggested values are "Yes", "No", or "Partial".

        :raises ValueError: If :code:`name` does not match the expected type.
        """
        self._name = name

        uri = BenchmarkUri.from_string(name)

        self._description = description
        self._license = license
        self._scheme = uri.scheme

        match = _DATASET_VERSION_RE.match(uri.dataset)
        self._version = int(match.group("version") if match else 0)
        self._references = references or {}
        self._deprecation_message = deprecated
        self._validatable = validatable

        self.sort_order = sort_order
        self.benchmark_class = benchmark_class

        # Set up the site data name.
        if site_data_base:
            self._site_data_path = (
                Path(site_data_base).resolve() / uri.scheme / uri.dataset
            )

    def __repr__(self):
        return self.name

    @property
    @mark_deprecated(
        version="0.2.1",
        reason=(
            "The `Dataset.logger` attribute is deprecated. All Dataset "
            "instances share a logger named compiler_gym.datasets"
        ),
    )
    def logger(self) -> logging.Logger:
        """The logger for this dataset.

        :type: logging.Logger
        """
        return _logger

    @property
    def name(self) -> str:
        """The name of the dataset.

        :type: str
        """
        return self._name

    @property
    def description(self) -> str:
        """A short human-readable description of the dataset.

        :type: str
        """
        return self._description

    @property
    def license(self) -> str:
        """The name of the license of the dataset.

        :type: str
        """
        return self._license

    @property
    @mark_deprecated(
        version="0.2.2", reason="The `protocol` attribute has been renamed `scheme`"
    )
    def protocol(self) -> str:
        """The URI scheme that is used to identify benchmarks in this dataset.

        :type: str
        """
        return self.scheme

    @property
    def scheme(self) -> str:
        """The URI scheme that is used to identify benchmarks in this dataset.

        :type: str
        """
        return self._scheme

    @property
    def version(self) -> int:
        """The version tag for this dataset. Defaults to zero.

        :type: int
        """
        return self._version

    @property
    def references(self) -> Dict[str, str]:
        """A dictionary of useful named URLs for this dataset containing extra
        information, download links, papers, etc.

        For example:

            >>> dataset.references
            {'Paper': 'https://arxiv.org/pdf/1407.3487.pdf',
            'Homepage': 'https://ctuning.org/wiki/index.php/CTools:CBench'}

        :type: Dict[str, str]
        """
        return self._references

    @property
    def deprecated(self) -> bool:
        """Whether the dataset is included in the iterable sequence of datasets
        of a containing :class:`Datasets <compiler_gym.datasets.Datasets>`
        collection.

        :type: bool
        """
        return self._deprecation_message is not None

    @property
    def validatable(self) -> str:
        """Whether the dataset is validatable. A validatable dataset is one
        where the behavior of the benchmarks can be checked by compiling the
        programs to binaries and executing them. If the benchmarks crash, or are
        found to have different behavior, then validation fails. This type of
        validation is used to check that the compiler has not broken the
        semantics of the program.

        This property takes a string and is used for documentation purposes
        only. Suggested values are "Yes", "No", or "Partial".

        :type: str
        """
        return self._validatable

    @property
    def has_site_data(self) -> bool:
        """Return whether the dataset has a site data directory.

        :type: bool
        """
        return hasattr(self, "_site_data_path")

    @property
    def site_data_path(self) -> Path:
        """The filesystem path used to store persistent dataset files.

        This directory may not exist.

        :type: Path

        :raises ValueError: If no site data path was specified at constructor
            time.
        """
        if not self.has_site_data:
            raise ValueError(f"Dataset has no site data path: {self.name}")
        return self._site_data_path

    @property
    def site_data_size_in_bytes(self) -> int:
        """The total size of the on-disk data used by this dataset.

        :type: int
        """
        if not self.has_site_data:
            return 0

        if not self.site_data_path.is_dir():
            return 0

        total_size = 0
        for dirname, _, filenames in os.walk(self.site_data_path):
            total_size += sum(
                os.path.getsize(os.path.join(dirname, f)) for f in filenames
            )
        return total_size

    @property
    def size(self) -> int:
        """The number of benchmarks in the dataset.

        If the number of benchmarks is unknown or unbounded, for example because
        the dataset represents a program generator that can produce an infinite
        number of programs, the value is 0.

        :type: int
        """
        return 0

    def __len__(self) -> int:
        """The number of benchmarks in the dataset.

        This is the same as :meth:`Dataset.size
        <compiler_gym.datasets.Dataset.size>`:

            >>> len(dataset) == dataset.size
            True

        If the number of benchmarks is unknown or unbounded, for example because
        the dataset represents a program generator that can produce an infinite
        number of programs, the value is 0.

        :return: An integer.
        """
        return self.size

    def __eq__(self, other: Union["Dataset", str]) -> bool:
        if isinstance(other, Dataset):
            return self.name == other.name
        return self.name == other

    def __lt__(self, other: Union["Dataset", str]) -> bool:
        if isinstance(other, Dataset):
            return self.name < other.name
        return self.name < other

    def __le__(self, other: Union["Dataset", str]) -> bool:
        return self < other or self == other

    @property
    def installed(self) -> bool:
        """Whether the dataset is installed locally. Installation occurs
        automatically on first use, or by calling :meth:`install()
        <compiler_gym.datasets.Dataset.install>`.

        :type: bool
        """
        return True

    def install(self) -> None:
        """Install this dataset locally.

        Implementing this method is optional. If implementing this method, you
        must call :code:`super().install()` first.

        This method should not perform redundant work. This method should first
        detect whether any work needs to be done so that repeated calls to
        :code:`install()` will complete quickly.
        """
        if self.deprecated:
            warnings.warn(
                f"Dataset '{self.name}' is marked as deprecated. {self._deprecation_message}",
                category=DeprecationWarning,
                stacklevel=2,
            )

    def uninstall(self) -> None:
        """Remove any local data for this benchmark.

        This method undoes the work of :meth:`install()
        <compiler_gym.datasets.Dataset.install>`. The dataset can still be used
        after calling this method.
        """
        if self.has_site_data() and self.site_data_path.is_dir():
            shutil.rmtree(self.site_data_path)

    def benchmarks(self) -> Iterable[Benchmark]:
        """Enumerate the (possibly infinite) benchmarks lazily.

        Iteration order is consistent across runs. The order of
        :meth:`benchmarks() <compiler_gym.datasets.Dataset.benchmarks>` and
        :meth:`benchmark_uris() <compiler_gym.datasets.Dataset.benchmark_uris>`
        is the same.

        If the number of benchmarks in the dataset is infinite
        (:code:`len(dataset) == math.inf`), the iterable returned by this method
        will continue indefinitely.

        :return: An iterable sequence of :class:`Benchmark
            <compiler_gym.datasets.Benchmark>` instances.
        """
        # Default implementation. Subclasses may wish to provide an alternative
        # implementation that is optimized to specific use cases.
        yield from (self.benchmark(uri) for uri in self.benchmark_uris())

    def __iter__(self) -> Iterable[Benchmark]:
        """Enumerate the (possibly infinite) benchmarks lazily.

        This is the same as :meth:`Dataset.benchmarks()
        <compiler_gym.datasets.Dataset.benchmarks>`:

            >>> from itertools import islice
            >>> list(islice(dataset, 100)) == list(islice(datset.benchmarks(), 100))
            True

        :return: An iterable sequence of :meth:`Benchmark
            <compiler_gym.datasets.Benchmark>` instances.
        """
        yield from self.benchmarks()

    def benchmark_uris(self) -> Iterable[str]:
        """Enumerate the (possibly infinite) benchmark URIs.

        Iteration order is consistent across runs. The order of
        :meth:`benchmarks() <compiler_gym.datasets.Dataset.benchmarks>` and
        :meth:`benchmark_uris() <compiler_gym.datasets.Dataset.benchmark_uris>`
        is the same.

        If the number of benchmarks in the dataset is infinite
        (:code:`len(dataset) == math.inf`), the iterable returned by this method
        will continue indefinitely.

        :return: An iterable sequence of benchmark URI strings.
        """
        raise NotImplementedError("abstract class")

    def benchmark_from_parsed_uri(self, uri: BenchmarkUri) -> Benchmark:
        """Select a benchmark.

        Subclasses must implement this method. Implementors may assume that the
        URI is well formed and that the :code:`scheme` and :code:`dataset`
        components are correct.

        :param uri: The parsed URI of the benchmark to return.

        :return: A :class:`Benchmark <compiler_gym.datasets.Benchmark>`
            instance.

        :raise LookupError: If :code:`uri` is not found.

        :raise ValueError: If the URI is invalid.
        """
        raise NotImplementedError("abstract class")

    def benchmark(self, uri: str) -> Benchmark:
        """Select a benchmark.

        :param uri: The URI of the benchmark to return.

        :return: A :class:`Benchmark <compiler_gym.datasets.Benchmark>`
            instance.

        :raise LookupError: If :code:`uri` is not found.

        :raise ValueError: If the URI is invalid.
        """
        return self.benchmark_from_parsed_uri(BenchmarkUri.from_string(uri))

    def random_benchmark(
        self, random_state: Optional[np.random.Generator] = None
    ) -> Benchmark:
        """Select a benchmark randomly.

        :param random_state: A random number generator. If not provided, a
            default :code:`np.random.default_rng()` is used.

        :return: A :class:`Benchmark <compiler_gym.datasets.Benchmark>`
            instance.
        """
        random_state = random_state or np.random.default_rng()
        return self._random_benchmark(random_state)

    def _random_benchmark(self, random_state: np.random.Generator) -> Benchmark:
        """Private implementation of the random benchmark getter.

        Subclasses must implement this method so that it selects a benchmark
        from the available benchmarks with uniform probability, using only
        :code:`random_state` as a source of randomness.
        """
        raise NotImplementedError("abstract class")

    def __getitem__(self, uri: str) -> Benchmark:
        """Select a benchmark by URI.

        This is the same as :meth:`Dataset.benchmark(uri)
        <compiler_gym.datasets.Dataset.benchmark>`:

            >>> dataset["benchmark://cbench-v1/crc32"] == dataset.benchmark("benchmark://cbench-v1/crc32")
            True

        :return: A :class:`Benchmark <compiler_gym.datasets.Benchmark>`
            instance.

        :raise LookupError: If :code:`uri` does not exist.
        """
        return self.benchmark(uri)


class DatasetInitError(OSError):
    """Base class for errors raised if a dataset fails to initialize."""


@mark_deprecated(
    version="0.1.4",
    reason=(
        "Datasets are now automatically activated. "
        "`More information <https://github.com/facebookresearch/CompilerGym/issues/45>`_."
    ),
)
def activate(env, dataset: Union[str, Dataset]) -> bool:
    """Deprecated function for managing datasets.

    :param dataset: The name of the dataset to download, or a :class:`Dataset
        <compiler_gym.datasets.Dataset>` instance.

    :return: :code:`True` if the dataset was activated, else :code:`False` if
        already active.

    :raises ValueError: If there is no dataset with that name.
    """
    return False


@mark_deprecated(
    version="0.1.4",
    reason=(
        "Please use :meth:`del env.datasets[dataset] <compiler_gym.datasets.Datasets.__delitem__>`. "
        "`More information <https://github.com/facebookresearch/CompilerGym/issues/45>`_."
    ),
)
def delete(env, dataset: Union[str, Dataset]) -> bool:
    """Deprecated function for managing datasets.

    Please use :meth:`del env.datasets[dataset]
    <compiler_gym.datasets.Datasets.__delitem__>`.

    :param dataset: The name of the dataset to download, or a :class:`Dataset
        <compiler_gym.datasets.Dataset>` instance.

    :return: :code:`True` if the dataset was deleted, else :code:`False` if
        already deleted.
    """
    del env.datasets[dataset]
    return False


@mark_deprecated(
    version="0.1.4",
    reason=(
        "Please use :meth:`env.datasets.deactivate() <compiler_gym.datasets.Datasets.deactivate>`. "
        "`More information <https://github.com/facebookresearch/CompilerGym/issues/45>`_."
    ),
)
def deactivate(env, dataset: Union[str, Dataset]) -> bool:
    """Deprecated function for managing datasets.

    Please use :meth:`del env.datasets[dataset]
    <compiler_gym.datasets.Datasets.__delitem__>`.

    :param dataset: The name of the dataset to download, or a :class:`Dataset
        <compiler_gym.datasets.Dataset>` instance.

    :return: :code:`True` if the dataset was deactivated, else :code:`False` if
        already inactive.
    """
    del env.datasets[dataset]
    return False


@mark_deprecated(
    version="0.1.7",
    reason=(
        "Datasets are now installed automatically, there is no need to call :code:`require()`. "
        "`More information <https://github.com/facebookresearch/CompilerGym/issues/45>`_."
    ),
)
def require(env, dataset: Union[str, Dataset]) -> bool:
    """Deprecated function for managing datasets.

    Datasets are now installed automatically. See :class:`env.datasets
        <compiler_gym.datasets.Datasets>`.

    :param env: The environment that this dataset is required for.

    :param dataset: The name of the dataset to download, or a :class:`Dataset
        <compiler_gym.datasets.Dataset>` instance.

    :return: :code:`True` if the dataset was downloaded, or :code:`False` if the
        dataset was already available.
    """
    return False
