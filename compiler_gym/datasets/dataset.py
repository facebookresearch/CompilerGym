# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import io
import json
import logging
import os
import shutil
import tarfile
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Union

import fasteners
from deprecated.sphinx import deprecated

from compiler_gym.datasets.benchmark import DATASET_NAME_RE, Benchmark
from compiler_gym.util.debug_util import get_logging_level
from compiler_gym.util.download import download


class Dataset(object):
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
        site_data_base: Path,
        benchmark_class=Benchmark,
        references: Optional[Dict[str, str]] = None,
        hidden: bool = False,
        sort_order: int = 0,
        logger: Optional[logging.Logger] = None,
        validatable: str = "No",
    ):
        """Constructor.

        :param name: The name of the dataset. Must conform to the pattern
            :code:`{{protocol}}://{{name}}-v{{version}}`.

        :param description: A short human-readable description of the dataset.

        :param license: The name of the dataset's license.

        :param site_data_base: The base path of a directory that will be used to
            store installed files.

        :param benchmark_class: The class to use when instantiating benchmarks.
            It must have the same constructor signature as :class:`Benchmark
            <compiler_gym.datasets.Benchmark>`.

        :param references: A dictionary containing URLs for this dataset, keyed
            by their name. E.g. :code:`references["Paper"] = "https://..."`.

        :param hidden: Whether the dataset should be excluded from the
            :meth:`datasets() <compiler_gym.datasets.Datasets.dataset>` iterator
            of any :class:`Datasets <compiler_gym.datasets.Datasets>` container.

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
        """
        self._name = name
        components = DATASET_NAME_RE.match(name)
        if not components:
            raise ValueError(
                f"Invalid dataset name: '{name}'. "
                "Dataset name must be in the form: '{{protocol}}://{{name}}-v{{version}}'"
            )
        self._description = description
        self._license = license
        self._protocol = components.group("dataset_protocol")
        self._version = int(components.group("dataset_version"))
        self._references = references or {}
        self._hidden = hidden
        self._validatable = validatable

        self._logger = logger
        self.sort_order = sort_order
        self.benchmark_class = benchmark_class

        # Set up the site data name.
        basename = components.group("dataset_name")
        self._site_data_path = Path(site_data_base).resolve() / self.protocol / basename

    def __repr__(self):
        return self.name

    @property
    def logger(self) -> logging.Logger:
        """The logger for this dataset.

        :type: logging.Logger
        """
        # NOTE(cummins): Default logger instantiation is deferred since in
        # Python 3.6 the Logger instance contains an un-pickle-able Rlock()
        # which can prevent gym.make() from working. This is a workaround that
        # can be removed once Python 3.6 support is dropped.
        if self._logger is None:
            self._logger = logging.getLogger("compiler_gym.datasets")
            self._logger.setLevel(get_logging_level())
        return self._logger

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
    def references(self) -> Dict[str, str]:
        """A dictionary containing URLs for this dataset, keyed by their name.
        E.g. :code:`references["Paper"] = "https://..."`.

        :type: Dict[str, str]
        """
        return self._references

    @property
    def hidden(self) -> str:
        """Whether the dataset is included in the iterable sequence of datasets
        of a containing :class:`Datasets <compiler_gym.datasets.Datasets>`
        collection.

        :type: bool
        """
        return self._hidden

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
    def site_data_path(self) -> Path:
        """The filesystem path used to store persistent dataset files.

        This directory may not exist.

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

    # We use Union[int, float] to represent the size because infinite size is
    # represented by math.inf, which is a float. For all other sizes this should
    # be an int.
    @property
    def size(self) -> Union[int, float]:
        """The number of benchmarks in the dataset. If the number of benchmarks
        is unbounded, for example because the dataset represents a program
        generator that can produce an infinite number of programs, the value is
        :code:`math.inf`.

        :type: Union[int, float]
        """
        return 0

    def __len__(self) -> Union[int, float]:
        """The number of benchmarks in the dataset.

        Equivalent to :meth:`Dataset.size <compiler_gym.datasets.Dataset.size>`.

        :return: An integer, or :code:`math.float`.
        """
        return self.size

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

        Equivalent to :meth:`Dataset.benchmarks()
        <compiler_gym.datasets.Dataset.benchmarks>`.

        :return: An iterable sequence of :meth:`Benchmark
            <compiler_gym.datasets.Benchmark>` instances.
        """
        yield from self.benchmarks()

    def benchmark_uris(self) -> Iterable[str]:
        """Enumerate the (possibly infinite) benchmark URIs.

        Benchmark URI order is consistent across runs. The order of
        :meth:`benchmarks() <compiler_gym.datasets.Dataset.benchmarks>` and
        :meth:`benchmark_uris() <compiler_gym.datasets.Dataset.benchmark_uris>`
        is the same.

        :return: An iterable sequence of benchmark URI strings.
        """
        raise NotImplementedError("abstract class")

    def benchmark(self, uri: str) -> Benchmark:
        """Select a benchmark.

        :param uri: The URI of the benchmark to return.

        :return: A :class:`Benchmark <compiler_gym.datasets.Benchmark>`
            instance.

        :raise LookupError: If :code:`uri` is not found.
        """
        raise NotImplementedError("abstract class")

    def __getitem__(self, uri: str) -> Benchmark:
        """Select a benchmark by URI.

        Equivalent to :meth:`Dataset.benchmark(uri)
        <compiler_gym.datasets.Dataset.benchmark>`.

        :return: A :class:`Benchmark <compiler_gym.datasets.Benchmark>`
            instance.

        :raise LookupError: If :code:`uri` does not exist.
        """
        return self.benchmark(uri)


class DatasetInitError(OSError):
    """Base class for errors raised if a dataset fails to initialize."""


class LegacyDataset(NamedTuple):
    """A collection of benchmarks for use by an environment.
    .. deprecated:: 0.1.4
       The next release of CompilerGym will introduce a new API for describing
       datasets with extended functionality. See
       `here <https://github.com/facebookresearch/CompilerGym/issues/45>`_ for
       more information.
    """

    name: str
    """The name of the dataset."""

    license: str
    """The license of the dataset."""

    file_count: int
    """The number of files in the unpacked dataset."""

    size_bytes: int
    """The size of the unpacked dataset in bytes."""

    url: str = ""
    """A URL where the dataset can be downloaded from. May be an empty string."""

    sha256: str = ""
    """The sha256 checksum of the dataset archive. If provided, this is used to
    verify the contents of the dataset upon download.
    """

    compiler: str = ""
    """The name of the compiler that this dataset supports."""

    description: str = ""
    """An optional human-readable description of the dataset."""

    platforms: List[str] = ["macos", "linux"]
    """A list of platforms supported by this dataset. Allowed platforms 'macos' and 'linux'."""

    deprecated_since: str = ""
    """The CompilerGym release in which this dataset was deprecated."""

    @property
    def deprecated(self) -> bool:
        """Whether the dataset is deprecated."""
        return bool(self.deprecated_since)

    @classmethod
    def from_json_file(cls, path: Path) -> "LegacyDataset":
        """Construct a dataset form a JSON metadata file.
        :param path: Path of the JSON metadata.
        :return: A LegacyDataset instance.
        """
        try:
            with open(str(path), "rb") as f:
                data = json.load(f)
        except json.decoder.JSONDecodeError as e:
            raise OSError(
                f"Failed to read dataset metadata file:\n"
                f"Path: {path}\n"
                f"Error: {e}"
            )
        return cls(**data)

    def to_json_file(self, path: Path) -> Path:
        """Write the dataset metadata to a JSON file.
        :param path: Path of the file to write.
        :return: The path of the written file.
        """
        with open(str(path), "wb") as f:
            json.dump(self._asdict(), f)
        return path


@deprecated(
    version="0.1.4",
    reason=(
        "Activating datasets will be removed in v0.1.5. "
        "`More information <https://github.com/facebookresearch/CompilerGym/issues/45>`_."
    ),
)
def activate(env, name: str) -> bool:
    """Move a directory from the inactive to active benchmark directory.
    :param: The name of a dataset.
    :return: :code:`True` if the dataset was activated, else :code:`False` if
        already active.
    :raises ValueError: If there is no dataset with that name.
    """
    with fasteners.InterProcessLock(env.datasets_site_path / "LOCK"):
        if (env.datasets_site_path / name).exists():
            # There is already an active benchmark set with this name.
            return False
        if not (env.inactive_datasets_site_path / name).exists():
            raise ValueError(f"Inactive dataset not found: {name}")
        os.rename(env.inactive_datasets_site_path / name, env.datasets_site_path / name)
        os.rename(
            env.inactive_datasets_site_path / f"{name}.json",
            env.datasets_site_path / f"{name}.json",
        )
        return True


@deprecated(
    version="0.1.4",
    reason=(
        "Deleting datasets will be removed in v0.1.5. "
        "`More information <https://github.com/facebookresearch/CompilerGym/issues/45>`_."
    ),
)
def delete(env, name: str) -> bool:
    """Delete a directory in the inactive benchmark directory.
    :param: The name of a dataset.
    :return: :code:`True` if the dataset was deleted, else :code:`False` if
        already deleted.
    """
    with fasteners.InterProcessLock(env.datasets_site_path / "LOCK"):
        deleted = False
        if (env.datasets_site_path / name).exists():
            shutil.rmtree(str(env.datasets_site_path / name))
            os.unlink(str(env.datasets_site_path / f"{name}.json"))
            deleted = True
        if (env.inactive_datasets_site_path / name).exists():
            shutil.rmtree(str(env.inactive_datasets_site_path / name))
            os.unlink(str(env.inactive_datasets_site_path / f"{name}.json"))
            deleted = True
        return deleted


@deprecated(
    version="0.1.4",
    reason=(
        "Deactivating datasets will be removed in v0.1.5. "
        "`More information <https://github.com/facebookresearch/CompilerGym/issues/45>`_."
    ),
)
def deactivate(env, name: str) -> bool:
    """Move a directory from active to the inactive benchmark directory.
    :param: The name of a dataset.
    :return: :code:`True` if the dataset was deactivated, else :code:`False` if
        already inactive.
    """
    with fasteners.InterProcessLock(env.datasets_site_path / "LOCK"):
        if not (env.datasets_site_path / name).exists():
            return False
        os.rename(env.datasets_site_path / name, env.inactive_datasets_site_path / name)
        os.rename(
            env.datasets_site_path / f"{name}.json",
            env.inactive_datasets_site_path / f"{name}.json",
        )
        return True


def require(env, dataset: Union[str, LegacyDataset]) -> bool:
    """Require that the given dataset is available to the environment.
    This will download and activate the dataset if it is not already installed.
    After calling this function, benchmarks from the dataset will be available
    to use.
    Example usage:
        >>> env = gym.make("llvm-v0")
        >>> require(env, "blas-v0")
        >>> env.reset(benchmark="blas-v0/1")
    :param env: The environment that this dataset is required for.
    :param dataset: The name of the dataset to download, the URL of the dataset,
        or a :class:`LegacyDataset` instance.
    :return: :code:`True` if the dataset was downloaded, or :code:`False` if the
        dataset was already available.
    """

    def download_and_unpack_archive(
        url: str, sha256: Optional[str] = None
    ) -> LegacyDataset:
        json_files_before = {
            f
            for f in env.inactive_datasets_site_path.iterdir()
            if f.is_file() and f.name.endswith(".json")
        }
        tar_data = io.BytesIO(download(url, sha256))
        with tarfile.open(fileobj=tar_data, mode="r:bz2") as arc:
            arc.extractall(str(env.inactive_datasets_site_path))
        json_files_after = {
            f
            for f in env.inactive_datasets_site_path.iterdir()
            if f.is_file() and f.name.endswith(".json")
        }
        new_json = json_files_after - json_files_before
        if not len(new_json):
            raise OSError(f"Downloaded dataset {url} contains no metadata JSON file")
        return LegacyDataset.from_json_file(list(new_json)[0])

    def unpack_local_archive(path: Path) -> LegacyDataset:
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        json_files_before = {
            f
            for f in env.inactive_datasets_site_path.iterdir()
            if f.is_file() and f.name.endswith(".json")
        }
        with tarfile.open(str(path), "r:bz2") as arc:
            arc.extractall(str(env.inactive_datasets_site_path))
        json_files_after = {
            f
            for f in env.inactive_datasets_site_path.iterdir()
            if f.is_file() and f.name.endswith(".json")
        }
        new_json = json_files_after - json_files_before
        if not len(new_json):
            raise OSError(f"Downloaded dataset {url} contains no metadata JSON file")
        return LegacyDataset.from_json_file(list(new_json)[0])

    with fasteners.InterProcessLock(env.datasets_site_path / "LOCK"):
        # Resolve the name and URL of the dataset.
        sha256 = None
        if isinstance(dataset, LegacyDataset):
            name, url = dataset.name, dataset.url
        elif isinstance(dataset, str):
            # Check if we have already downloaded the dataset.
            if "://" in dataset:
                name, url = None, dataset
                dataset: Optional[LegacyDataset] = None
            else:
                try:
                    dataset: Optional[LegacyDataset] = env.available_datasets[dataset]
                except KeyError:
                    raise ValueError(f"Dataset not found: {dataset}")
                name, url, sha256 = dataset.name, dataset.url, dataset.sha256
        else:
            raise TypeError(
                f"require() called with unsupported type: {type(dataset).__name__}"
            )

        if dataset and dataset.deprecated:
            warnings.warn(
                f"Dataset '{dataset.name}' is deprecated as of CompilerGym "
                f"release {dataset.deprecated_since}, please update to the "
                "latest available version",
                DeprecationWarning,
            )

        # Check if we have already downloaded the dataset.
        if name:
            if (env.datasets_site_path / name).is_dir():
                # Dataset is already downloaded and active.
                return False
            elif not (env.inactive_datasets_site_path / name).is_dir():
                # Dataset is downloaded but inactive.
                name = download_and_unpack_archive(url, sha256=sha256).name
        elif url.startswith("file:///"):
            name = unpack_local_archive(Path(url[len("file:///") :])).name
        else:
            name = download_and_unpack_archive(url, sha256=sha256).name

        activate(env, name)
        return True
