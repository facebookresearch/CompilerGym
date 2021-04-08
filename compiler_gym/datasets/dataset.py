# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
from deprecated.sphinx import deprecated

from compiler_gym.datasets.benchmark import Benchmark
from compiler_gym.util.debug_util import get_logging_level

# Regular expression that matches the full three-part format of a benchmark URI:
#     <protocol>://<dataset>/<id>
#
# E.g. "benchmark://foo-v0/" or "benchmark://foo-v0/program".
DATASET_NAME_RE = re.compile(
    r"(?P<dataset>(?P<dataset_protocol>[a-zA-z0-9-_]+)://(?P<dataset_name>[a-zA-z0-9-_]+-v(?P<dataset_version>[0-9]+)))"
)

BENCHMARK_URI_RE = re.compile(
    r"(?P<dataset>(?P<dataset_protocol>[a-zA-z0-9-_]+)://(?P<dataset_name>[a-zA-z0-9-_]+-v(?P<dataset_version>[0-9]+)))/(?P<benchmark_name>[^\s]*)$"
)


class Dataset(object):
    """A dataset is a collection of benchmarks.

    Datasets provide a convenience abstraction for installing and managing
    groups of benchmarks. Datasets provide the API for instantiating benchmarks,
    either by randomly selecting from the available benchmarks or by selecting
    by URI.

    The Dataset class is an abstract base for implementing datasets. At a
    minimum, subclasses must implement the :meth:`benchmark()
    <compiler_gym.datasets.Dataset.benchmark>` and :meth:`benchmark_uris()
    <compiler_gym.datasets.Dataset.benchmark_uris>` methods. Other methods such
    as :meth:`install() <compiler_gym.datasets.Dataset.install>` may be used
    where helpful.
    """

    def __init__(
        self,
        name: str,
        description: str,
        license: str,
        site_data_base: Path,
        benchmark_class=Benchmark,
        long_description_url: Optional[str] = None,
        random: Optional[np.random.Generator] = None,
        hidden: bool = False,
        sort_order: int = 0,
    ):
        """Constructor.

        :param name: The name of the dataset.

        :param description: A short human-readable description of the dataset.

        :param license: The name of the dataset's license.

        :param site_data_base: The base path of a directory that will be used to
            store installed files.

        :param benchmark_class: The class to use when instantiating benchmarks.
            It must have the same constructor signature as :class:`Benchmark
            <compiler_gym.datasets.Benchmark>`.

        :param long_description_url: The URL of a website that contains a more
            detailed description of the dataset.

        :param random: A source of randomness for selecting benchmarks.

        :param hidden: Whether the dataset should be excluded from the
            :meth:`datasets() <compiler_gym.datasets.Datasets.dataset>` iterator
            of any :class:`Datasets <compiler_gym.datasets.Datasets>` container.

        :param sort_order: An optional numeric value that should be used to
            order this dataset relative to others. Lowest value sorts first.
        """
        self._name = name
        components = DATASET_NAME_RE.match(name)
        if not components:
            raise ValueError(
                f"Invalid dataset name: '{name}'. "
                "Dataset name must be in the form: '${protocol}://${name}-v${version}'"
            )
        self._description = description
        self._license = license
        self._protocol = components.group("dataset_protocol")
        self._version = int(components.group("dataset_version"))
        self._long_description_url = long_description_url
        self._hidden = hidden

        self.random = random or np.random.default_rng()
        self.logger = logging.getLogger("compiler_gym.datasets")
        self.logger.setLevel(get_logging_level())
        self.sort_order = sort_order
        self.benchmark_class = benchmark_class

        # Set up the site data name.
        basename = components.group("dataset_name")
        self._site_data_path = Path(site_data_base).resolve() / self.protocol / basename

    def __repr__(self):
        return self.name

    def seed(self, seed: int):
        """Set the random state.

        Setting a random state will fix the order that
        :meth:`dataset.benchmark() <compiler_gym.datasets.Dataset.benchmark>`
        returns benchmarks when called without arguments.

        :param seed: A number.
        """
        self.random = np.random.default_rng(seed)

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
    def protocol(self) -> str:
        """The URI protocol that is used to identify benchmarks in this dataset.

        :type: str
        """
        return self._protocol

    @property
    def version(self) -> int:
        """A version tag for this dataset.

        :type: int
        """
        return self._version

    @property
    def long_description_url(self) -> str:
        """The URL of a website with further information about this dataset.

        :type: str
        """
        return self._long_description_url

    @property
    def hidden(self) -> str:
        """Whether the dataset is included in the iterable sequence of datasets
        of a containing :class:`Datasets <compiler_gym.datasets.Datasets>`
        collection.

        :type: bool
        """
        return self._hidden

    @property
    def site_data_path(self) -> Path:
        """The filesystem path used to store persistent dataset files.

        :type: Path
        """
        return self._site_data_path

    @property
    def site_data_size_in_bytes(self) -> int:
        """The total size of the on-disk data used by this dataset.

        :type: int
        """
        if not self.site_data_path.is_dir():
            return 0

        total_size = 0
        for dirname, _, filenames in os.walk(self.site_data_path):
            total_size += sum(
                os.path.getsize(os.path.join(dirname, f)) for f in filenames
            )
        return total_size

    @property
    def n(self) -> int:
        """The number of benchmarks in the dataset. This value is negative if
        the number of benchmarks in the dataset is unbounded, for example
        because the dataset represents a program generator that can produce an
        infinite number of programs.

        :type: int
        """
        return 0

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

        Implementing this method is optional.

        This method should not perform redundant work - it should detect whether
        any work needs to be done so that repeated calls to install will
        complete quickly.
        """
        pass

    def uninstall(self) -> None:
        """Remove any local data for this benchmark.

        The dataset can still be used after calling this method. This method
        just undoes the work of :meth:`install()
        <compiler_gym.datasets.Dataset.install>`.
        """
        if self.site_data_path.is_dir():
            shutil.rmtree(self.site_data_path)

    def benchmarks(self) -> Iterable[Benchmark]:
        """Enumerate the (possibly infinite) benchmarks lazily.

        Benchmark order is consistent across runs. The order of
        :meth:`benchmarks() <compiler_gym.datasets.Dataset.benchmarks>` and
        :meth:`benchmark_uris() <compiler_gym.datasets.Dataset.benchmark_uris>`
        is the same.

        :return: An iterable sequence of :class:`Benchmark
            <compiler_gym.datasets.Benchmark>` instances.
        """
        # Default implementation. Subclasses may wish to provide an alternative
        # implementation that is optimized to specific use cases.
        yield from (self.benchmark(uri) for uri in self.benchmark_uris())

    def benchmark_uris(self) -> Iterable[str]:
        """Enumerate the (possibly infinite) benchmark URIs.

        Benchmark URI order is consistent across runs. The order of
        :meth:`benchmarks() <compiler_gym.datasets.Dataset.benchmarks>` and
        :meth:`benchmark_uris() <compiler_gym.datasets.Dataset.benchmark_uris>`
        is the same.

        :return: An iterable sequence of benchmark URI strings.
        """
        raise NotImplementedError("abstract class")

    def benchmark(self, uri: Optional[str] = None) -> Benchmark:
        """Select a benchmark.

        If a URI is given, the corresponding :class:`Benchamrk
        <compiler_gym.datasets.Benchmark>` is returned. Otherwise, a benchmark
        is selected uniformly randomly.

        Use :meth:`seed() <compiler_gym.datasets.Dataset.seed>` to force a
        reproducible order for randomly selected benchmarks.

        :param uri: The URI of the benchmark to return. If :code:`None`, select
            a benchmark randomly using :code:`self.random`.

        :return: A :class:`Benchamrk <compiler_gym.datasets.Benchmark>`
            instance.

        :raise LookupError: If :code:`uri` is provided but does not exist.
        """
        raise NotImplementedError("abstract class")


@deprecated(
    version="0.1.4",
    reason=(
        "Please use :meth:`env.datasets.activate() <compiler_gym.datasets.Datasets.activate>`. "
        "`More information <https://github.com/facebookresearch/CompilerGym/issues/45>`_."
    ),
)
def activate(env, dataset: Union[str, Dataset]) -> bool:
    """Activate a dataset.

    Deprecated. Use :meth:`datasets.activate()
    <compiler_gym.datasets.Datasets.activate>`.

    :param dataset: The name of the dataset to download, or a :class:`Dataset`
        instance.

    :return: :code:`True` if the dataset was activated, else :code:`False` if
        already active.

    :raises ValueError: If there is no dataset with that name.
    """
    return env.datasets.activate(dataset)


@deprecated(
    version="0.1.4",
    reason=(
        "Please use :meth:`env.datasets.remove() <compiler_gym.datasets.Datasets.remove>`. "
        "`More information <https://github.com/facebookresearch/CompilerGym/issues/45>`_."
    ),
)
def delete(env, dataset: Union[str, Dataset]) -> bool:
    """Remove a dataset.

    Deprecated. Use :meth:`datasets.remove()
    <compiler_gym.datasets.Datasets.remove>`.

    :param dataset: The name of the dataset to download, or a :class:`Dataset`
        instance.

    :return: :code:`True` if the dataset was deleted, else :code:`False` if
        already deleted.
    """
    return env.datasets.remove(dataset)


@deprecated(
    version="0.1.4",
    reason=(
        "Please use :meth:`env.datasets.deactivate() <compiler_gym.datasets.Datasets.deactivate>`. "
        "`More information <https://github.com/facebookresearch/CompilerGym/issues/45>`_."
    ),
)
def deactivate(env, dataset: Union[str, Dataset]) -> bool:
    """Deactivate a dataset.

    Deprecated. Use :meth:`datasets.deactivate()
    <compiler_gym.datasets.Datasets.deactivate>`.

    :param dataset: The name of the dataset to download, or a :class:`Dataset`
        instance.

    :return: :code:`True` if the dataset was deactivated, else :code:`False` if
        already inactive.
    """
    return env.datasets.deactivate(dataset)


@deprecated(
    version="0.1.7",
    reason=(
        "Please use :meth:`env.datasets.require() <compiler_gym.datasets.Datasets.require>`. "
        "`More information <https://github.com/facebookresearch/CompilerGym/issues/45>`_."
    ),
)
def require(env, dataset: Union[str, Dataset]) -> bool:
    """Require that the given dataset is installed in the environment.

    This will download and activate the dataset if it is not already installed.
    After calling this function, benchmarks from the dataset will be available
    to use.

    Example usage:

        >>> env = gym.make("llvm-v0")
        >>> require(env, "blas-v0")
        >>> env.reset(benchmark="blas-v0/1")

    :param env: The environment that this dataset is required for.

    :param dataset: The name of the dataset to download, or a :class:`Dataset`
        instance.

    :return: :code:`True` if the dataset was downloaded, or :code:`False` if the
        dataset was already available.
    """
    if isinstance(dataset, Dataset):
        dataset.install()
    else:
        env.datasets.dataset(dataset).install()
    return True
