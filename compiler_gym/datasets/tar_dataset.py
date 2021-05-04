# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import bz2
import gzip
import io
import shutil
import tarfile
from threading import Lock
from typing import Iterable, List, Optional

from fasteners import InterProcessLock

from compiler_gym.datasets.files_dataset import FilesDataset
from compiler_gym.util.decorators import memoized_property
from compiler_gym.util.download import download
from compiler_gym.util.filesystem import atomic_file_write


class TarDataset(FilesDataset):
    """A dataset comprising a files tree stored in a tar archive.

    This extends the :class:`FilesDataset <compiler_gym.datasets.FilesDataset>`
    class by adding support for compressed archives of files. The archive is
    downloaded and unpacked on-demand.
    """

    def __init__(
        self,
        tar_urls: List[str],
        tar_sha256: Optional[str] = None,
        tar_compression: str = "bz2",
        strip_prefix: str = "",
        **dataset_args,
    ):
        """Constructor.

        :param tar_urls: A list of redundant URLS to download the tar archive from.

        :param tar_sha256: The SHA256 checksum of the downloaded tar archive.

        :param tar_compression: The tar archive compression type. One of
            {"bz2", "gz"}.

        :param strip_prefix: An optional path prefix to strip. Only files that
            match this path prefix will be used as benchmarks.

        :param dataset_args: See :meth:`FilesDataset.__init__()
            <compiler_gym.datasets.FilesDataset.__init__>`.
        """
        super().__init__(
            dataset_root=None,  # Set below once site_data_path is resolved.
            **dataset_args,
        )
        self.dataset_root = self.site_data_path / "contents" / strip_prefix

        self.tar_urls = tar_urls
        self.tar_sha256 = tar_sha256
        self.tar_compression = tar_compression
        self.strip_prefix = strip_prefix

        self._tar_extracted_marker = self.site_data_path / ".extracted"
        self._tar_lock = Lock()
        self._tar_lockfile = self.site_data_path / ".install_lock"

    @property
    def installed(self) -> bool:
        return self._tar_extracted_marker.is_file()

    def install(self) -> None:
        super().install()

        if self.installed:
            return

        # Thread-level and process-level locks to prevent races.
        with self._tar_lock, InterProcessLock(self._tar_lockfile):
            # Repeat the check to see if we have already installed the
            # dataset now that we have acquired the lock.
            if self.installed:
                return

            # Remove any partially-completed prior extraction.
            shutil.rmtree(self.site_data_path / "contents", ignore_errors=True)

            self.logger.info("Downloading %s dataset", self.name)
            tar_data = io.BytesIO(download(self.tar_urls, self.tar_sha256))
            self.logger.info("Unpacking %s dataset", self.name)
            with tarfile.open(
                fileobj=tar_data, mode=f"r:{self.tar_compression}"
            ) as arc:
                arc.extractall(str(self.site_data_path / "contents"))

            # We're done. The last thing we do is create the marker file to
            # signal to any other install() invocations that the dataset is
            # ready.
            self._tar_extracted_marker.touch()

        if self.strip_prefix and not self.dataset_root.is_dir():
            raise FileNotFoundError(
                f"Directory prefix '{self.strip_prefix}' not found in dataset '{self.name}'"
            )


class TarDatasetWithManifest(TarDataset):
    """A tarball-based dataset that reads the benchmark URIs from a separate
    manifest file.

    A manifest file is a plain text file containing a list of benchmark names,
    one per line, and is shipped separately from the tar file. The idea is to
    allow the list of benchmark URIs to be enumerated in a more lightweight
    manner than downloading and unpacking the entire dataset. It does this by
    downloading and unpacking only the manifest to iterate over the URIs.

    The manifest file is assumed to be correct and is not validated.
    """

    def __init__(
        self,
        manifest_urls: List[str],
        manifest_sha256: str,
        manifest_compression: str = "bz2",
        **dataset_args,
    ):
        """Constructor.

        :param manifest_urls: A list of redundant URLS to download the
            compressed text file containing a list of benchmark URI suffixes,
            one per line.

        :param manifest_sha256: The sha256 checksum of the compressed manifest
            file.

        :param manifest_compression: The manifest compression type. One of
            {"bz2", "gz"}.

        :param dataset_args: See :meth:`TarDataset.__init__()
            <compiler_gym.datasets.TarDataset.__init__>`.
        """
        super().__init__(**dataset_args)
        self.manifest_urls = manifest_urls
        self.manifest_sha256 = manifest_sha256
        self.manifest_compression = manifest_compression
        self._manifest_path = self.site_data_path / f"manifest-{manifest_sha256}.txt"

        self._manifest_lock = Lock()
        self._manifest_lockfile = self.site_data_path / ".manifest_lock"

    def _read_manifest(self, manifest_data: str) -> List[str]:
        """Read the manifest data into a list of URIs. Does not validate the
        manifest contents.
        """
        lines = manifest_data.rstrip().split("\n")
        return [f"{self.name}/{line}" for line in lines]

    def _read_manifest_file(self) -> List[str]:
        """Read the benchmark URIs from an on-disk manifest file.

        Does not check that the manifest file exists.
        """
        with open(self._manifest_path, encoding="utf-8") as f:
            uris = self._read_manifest(f.read())
        self.logger.debug("Read %s manifest, %d entries", self.name, len(uris))
        return uris

    @memoized_property
    def _benchmark_uris(self) -> List[str]:
        """Fetch or download the URI list."""
        if self._manifest_path.is_file():
            return self._read_manifest_file()

        # Thread-level and process-level locks to prevent races.
        with self._manifest_lock, InterProcessLock(self._manifest_lockfile):
            # Now that we have acquired the lock, repeat the check, since
            # another thread may have downloaded the manifest.
            if self._manifest_path.is_file():
                return self._read_manifest_file()

            # Determine how to decompress the manifest data.
            decompressor = {
                "bz2": lambda compressed_data: bz2.BZ2File(compressed_data),
                "gz": lambda compressed_data: gzip.GzipFile(compressed_data),
            }.get(self.manifest_compression, None)
            if not decompressor:
                raise TypeError(
                    f"Unknown manifest compression: {self.manifest_compression}"
                )

            # Decompress the manifest data.
            self.logger.debug("Downloading %s manifest", self.name)
            manifest_data = io.BytesIO(
                download(self.manifest_urls, self.manifest_sha256)
            )
            with decompressor(manifest_data) as f:
                manifest_data = f.read()

            # Although we have exclusive-execution locks, we still need to
            # create the manifest atomically to prevent calls to _benchmark_uris
            # racing to read an incompletely written manifest.
            with atomic_file_write(self._manifest_path, fileobj=True) as f:
                f.write(manifest_data)

            uris = self._read_manifest(manifest_data.decode("utf-8"))
            self.logger.debug(
                "Downloaded %s manifest, %d entries", self.name, len(uris)
            )
            return uris

    @memoized_property
    def size(self) -> int:
        return len(self._benchmark_uris)

    def benchmark_uris(self) -> Iterable[str]:
        yield from iter(self._benchmark_uris)
