# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from pathlib import Path
from typing import Iterable, List

import numpy as np

from compiler_gym.datasets.dataset import Benchmark, Dataset
from compiler_gym.util.decorators import memoized_property


class FilesDataset(Dataset):
    """A dataset comprising a directory tree of files.

    A FilesDataset is a root directory that contains (a possibly nested tree of)
    files, where each file represents a benchmark. The directory contents can be
    filtered by specifying a filename suffix that files must match.

    The URI of benchmarks is the relative path of each file, stripped of a
    required filename suffix, if specified. For example, given the following
    file tree:

    .. code-block::

        /tmp/dataset/a.txt
        /tmp/dataset/LICENSE
        /tmp/dataset/subdir/subdir/b.txt
        /tmp/dataset/subdir/subdir/c.txt

    a FilesDataset :code:`benchmark://ds-v0` rooted at :code:`/tmp/dataset` with
    filename suffix :code:`.txt` will contain the following URIs:

        >>> list(dataset.benchmark_uris())
        [
            "benchmark://ds-v0/a",
            "benchmark://ds-v0/subdir/subdir/b",
            "benchmark://ds-v0/subdir/subdir/c",
        ]
    """

    def __init__(
        self,
        dataset_root: Path,
        benchmark_file_suffix: str = "",
        memoize_uris: bool = True,
        **dataset_args,
    ):
        """Constructor.

        :param dataset_root: The root directory to look for benchmark files.

        :param benchmark_file_suffix: A file extension that must be matched for
            a file to be used as a benchmark.

        :param memoize_uris: Whether to memoize the list of URIs contained in
            the dataset. Memoizing the URIs enables faster repeated iteration
            over :meth:`dataset.benchmark_uris()
            <compiler_gym.datasets.Dataset.benchmark_uris>` at the expense of
            increased memory overhead as the file list must be kept in memory.

        :param dataset_args: See :meth:`Dataset.__init__()
            <compiler_gym.datasets.Dataset.__init__>`.
        """
        super().__init__(**dataset_args)
        self.dataset_root = dataset_root
        self.benchmark_file_suffix = benchmark_file_suffix
        self.memoize_uris = memoize_uris
        self._memoized_uris = None

    @memoized_property
    def size(self) -> int:  # pylint: disable=invalid-overriden-method
        self.install()
        return sum(
            sum(1 for f in files if f.endswith(self.benchmark_file_suffix))
            for (_, _, files) in os.walk(self.dataset_root)
        )

    @property
    def _benchmark_uris_iter(self) -> Iterable[str]:
        """Return an iterator over benchmark URIs that is consistent across runs."""
        self.install()
        for root, dirs, files in os.walk(self.dataset_root):
            # Sort the subdirectories so that os.walk() order is stable between
            # runs.
            dirs.sort()
            reldir = root[len(str(self.dataset_root)) + 1 :]
            for filename in sorted(files):
                # If we have an expected file suffix then ignore files that do
                # not match, and strip the suffix from files that do match.
                if self.benchmark_file_suffix:
                    if not filename.endswith(self.benchmark_file_suffix):
                        continue
                    filename = filename[: -len(self.benchmark_file_suffix)]
                # Use os.path.join() rather than simple '/' concatenation as
                # reldir may be empty.
                yield os.path.join(self.name, reldir, filename)

    @property
    def _benchmark_uris(self) -> List[str]:
        return list(self._benchmark_uris_iter)

    def benchmark_uris(self) -> Iterable[str]:
        if self._memoized_uris:
            yield from self._memoized_uris
        elif self.memoize_uris:
            self._memoized_uris = self._benchmark_uris
            yield from self._memoized_uris
        else:
            yield from self._benchmark_uris_iter

    def benchmark(self, uri: str) -> Benchmark:
        self.install()

        relpath = f"{uri[len(self.name) + 1:]}{self.benchmark_file_suffix}"
        abspath = self.dataset_root / relpath
        if not abspath.is_file():
            raise LookupError(f"Benchmark not found: {uri} (file not found: {abspath})")
        return self.benchmark_class.from_file(uri, abspath)

    def _random_benchmark(self, random_state: np.random.Generator) -> Benchmark:
        return self.benchmark(random_state.choice(list(self.benchmark_uris())))
