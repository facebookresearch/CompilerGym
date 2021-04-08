# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import io
import os
import shutil
import tarfile
from gzip import GzipFile
from threading import Lock
from typing import Iterable, List, Optional

import fasteners

from compiler_gym.datasets.files_dataset import FilesDataset
from compiler_gym.util.decorators import memoized_property
from compiler_gym.util.download import download


class TarDataset(FilesDataset):
    """A dataset comprising a files tree stored in a tar archive.

    This extends the :class:`FilesDataset` class by adding support for
    compressed archives of files. The archive is downloaded and unpacked
    on-demand.
    """

    def __init__(
        self,
        tar_url: str,
        tar_sha256: Optional[str] = None,
        tar_compression: str = "bz2",
        strip_prefix: str = "",
        **dataset_args,
    ):
        """Constructor.

        :param tar_url: The URL of the tar archive to download.

        :param tar_sha256: The SHA256 checksum of the downloaded tar archive.

        :param tar_compression: The tar archive compression type.

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

        self.tar_url = tar_url
        self.tar_sha256 = tar_sha256
        self.tar_compression = tar_compression
        self.strip_prefix = strip_prefix

        self._installed = False
        self._tar_extracted_marker = self.site_data_path / ".extracted"
        self._tar_lock = Lock()
        self._tar_lockfile = self.site_data_path / "LOCK"

    @property
    def installed(self) -> bool:
        # Fast path for repeated checks to 'installed' without a disk op.
        if not self._installed:
            self._installed = self._tar_extracted_marker.is_file()
        return self._installed

    def install(self) -> None:
        if self.installed:
            return

        # Thread-level and process-level locks to prevent races.
        with self._tar_lock:
            with fasteners.InterProcessLock(self._tar_lockfile):
                # Repeat the check to see if we have already installed the
                # dataset now that we have acquired the lock, else there will be
                # installation races.
                if self._tar_extracted_marker.is_file():
                    return

                self.logger.info("Downloading %s dataset", self.name)
                tar_data = io.BytesIO(download(self.tar_url, self.tar_sha256))
                with tarfile.open(
                    fileobj=tar_data, mode=f"r:{self.tar_compression}"
                ) as arc:
                    # Remove any partially-completed prior extraction.
                    shutil.rmtree(self.site_data_path / "contents", ignore_errors=True)

                    arc.extractall(str(self.site_data_path / "contents"))

                self._tar_extracted_marker.touch()

        if self.strip_prefix and not self.dataset_root.is_dir():
            raise FileNotFoundError(
                f"Directory prefix '{self.strip_prefix}' not found in dataset '{self.name}'"
            )


class TarDatasetWithManifest(TarDataset):
    """A tarball-based dataset that has a manifest file which lists the URIs.

    The idea is that this allows enumerating the list of available datasets in a
    more lightweight manner than downloading and unpacking the entire dataset.

    The manifest is assumed to be correct and is not validated.
    """

    def __init__(self, manifest_url: str, manifest_sha256: str, **dataset_args):
        """Constructor.

        :param manifest_url: The URL of a gzip-compressed text file containing a
            list of benchmark URIs, one per line.

        :param manifest_sha256: The sha256 checksum of the compressed manifest
            file.

        :param dataset_args: See :meth:`TarDataset.__init__()
            <compiler_gym.datasets.TarDataset.__init__>`.
        """
        super().__init__(**dataset_args)
        self.manifest_url = manifest_url
        self.manifest_sha256 = manifest_sha256
        self._manifest_path = self.site_data_path / f"manifest-{manifest_sha256}.txt"

        self._manifest_lock = Lock()
        self._manifest_lockfile = self.site_data_path / "manifest.LOCK"

    def _read_manifest_file(self) -> List[str]:
        with open(self._manifest_path) as f:
            uris = f.read().rstrip().split("\n")
            self.logger.debug("Read %s manifest, %d entries", self.name, len(uris))
        return uris

    @memoized_property
    def _benchmark_uris(self) -> List[str]:
        if self._manifest_path.is_file():
            return self._read_manifest_file()

        with self._manifest_lock:
            with fasteners.InterProcessLock(self._manifest_lockfile):
                # Now that we have acquired the lock, repeat the check.
                if self._manifest_path.is_file():
                    return self._read_manifest_file()

                self.logger.debug("Downloading %s manifest", self.name)
                manifest_data = io.BytesIO(
                    download(self.manifest_url, self.manifest_sha256)
                )
                with GzipFile(fileobj=manifest_data) as gzipf:
                    manifest_data = gzipf.read()

                # Write to a temporary file and rename to avoid a race.
                with open(f"{self._manifest_path}.tmp", "wb") as f:
                    f.write(manifest_data)
                os.rename(f"{self._manifest_path}.tmp", self._manifest_path)

                uris = manifest_data.decode("utf-8").rstrip().split("\n")
                self.logger.debug(
                    "Downloaded %s manifest, %d entries", self.name, len(uris)
                )
                return uris

    @memoized_property
    def n(self) -> int:
        return len(self._benchmark_uris)

    def benchmark_uris(self) -> Iterable[str]:
        yield from iter(self._benchmark_uris)
