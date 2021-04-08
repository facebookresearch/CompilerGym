# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from pathlib import Path
from typing import Iterable, List, Optional

from compiler_gym.datasets.dataset import Benchmark, Dataset
from compiler_gym.util.decorators import memoized_property


class FilesDataset(Dataset):
    """A dataset comprising a directory tree files.

    A FilesDataset is a root directory that contains (possibly nested) files,
    where each file represents a benchmark. Files can be filtered on their
    expected filename suffix.

    Every file that matches the expected suffix is a benchmark. The URI of the
    benchmarks is the relative path of the file within the dataset, stripped of
    any matching file suffix. For example, given a dataset root around the
    directory tree :code:`/tmp/dataset` and with file suffix :code:`.txt`:

    .. code-block::

        /tmp/dataset/a.txt
        /tmp/dataset/LICENSE
        /tmp/dataset/subdir/subdir/b.txt
        /tmp/dataset/subdir/subdir/c.txt

    a FilesDataset representing this directory tree will contain the following
    URIs:

        >>> list(dataset.benchmark_uris())
        [
            "benchamrk://ds-v0/a",
            "benchamrk://ds-v0/subdir/subdir/b",
            "benchamrk://ds-v0/subdir/subdir/c",
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
            the dataset. Memoizing the URIs is a tradeoff between *O(n)*
            computation complexity of random access vs *O(n)* space complexity
            of memoizing the URI list.

        :param dataset_args: See :meth:`Dataset.__init__()
            <compiler_gym.datasets.Dataset.__init__>`.
        """
        super().__init__(**dataset_args)
        self.dataset_root = dataset_root
        self.benchmark_file_suffix = benchmark_file_suffix
        self.memoize_uris = memoize_uris
        self._memoized_uris = None

    @memoized_property
    def n(self) -> int:  # pylint: disable=invalid-overriden-method
        self.install()
        return sum(
            sum(1 for f in files if f.endswith(self.benchmark_file_suffix))
            for (_, _, files) in os.walk(self.dataset_root)
        )

    @property
    def _benchmark_uris_iter(self) -> Iterable[str]:
        """Return an iterator over benchmark URIs is consistent across runs."""
        self.install()
        for root, dirs, files in os.walk(self.dataset_root):
            dirs.sort()
            reldir = root[len(str(self.dataset_root)) + 1 :]
            for filename in sorted(files):
                # If we have an expected file suffix then filter on it and strip
                # it from the filename.
                if self.benchmark_file_suffix:
                    if not filename.endswith(self.benchmark_file_suffix):
                        continue
                    filename = filename[: -len(self.benchmark_file_suffix)]
                # Use os.path.join() rather than simple '/' concaentation as
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

    def benchmark(self, uri: Optional[str] = None) -> Benchmark:
        self.install()
        if uri is None:
            if not self.n:
                raise ValueError("No benchmarks")
            return self.get_benchmark_by_index(self.random.integers(self.n))

        relpath = f"{uri[len(self.name) + 1:]}{self.benchmark_file_suffix}"
        abspath = self.dataset_root / relpath
        if not abspath.is_file():
            raise LookupError(f"Benchmark not found: {uri} (file not found: {abspath})")
        return self.benchmark_class.from_file(uri, abspath)

    def get_benchmark_by_index(self, n: int) -> Benchmark:
        """Look a benchmark using a numeric index into the list of URIs."""
        # If we have memoized the benchmark IDs then just index into the list.
        # Otherwise we will scan through the directory hierarchy.
        if self.memoize_uris:
            if not self._memoized_uris:
                self._memoized_uris = self._benchmark_uris
            return self.benchmark(self._memoized_uris[n])

        i = 0
        for root, dirs, files in os.walk(self.dataset_root):
            reldir = root[len(str(self.dataset_root)) + 1 :]

            # Filter only the files that match the target file suffix.
            valid_files = [f for f in files if f.endswith(self.benchmark_file_suffix)]

            if i + len(valid_files) < n:
                # There aren't enough files in this directory to bring us up to
                # the target file index, so skip the whole lot.
                i += len(valid_files)
                dirs.sort()
            else:
                # Iterate through the files in the directory in order until we
                # reach the target index.
                for filename in sorted(valid_files):
                    if i == n:
                        name_stem = filename[: -len(self.benchmark_file_suffix)]
                        # Use os.path.join() rather than simple '/' concaentation as
                        # reldir may be empty.
                        uri = os.path.join(self.name, reldir, name_stem)
                        return self.benchmark_class.from_file(uri, f"{root}/{filename}")
                    i += 1

        raise FileNotFoundError(f"Could not find benchmark with index {n} / {self.n}")
