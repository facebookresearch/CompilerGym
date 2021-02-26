# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Module for resolving paths to LLVM binaries and libraries."""
import io
import shutil
import sys
import tarfile
from pathlib import Path
from threading import Lock

import fasteners

from compiler_gym.util.download import download
from compiler_gym.util.runfiles_path import cache_path, site_data_path

# (url, sha256) tuples for the LLVM download data packs.
_LLVM_URLS = {
    "darwin": (
        "https://dl.fbaipublicfiles.com/compiler_gym/llvm-10.0.0-v0-macos.tar.bz2",
        "ab89ccfe3841b16251deb495af68dcd121fec39f70c67281e521d8cc331b9d71",
    ),
    "linux": (
        "https://dl.fbaipublicfiles.com/compiler_gym/llvm-10.0.0-v0-linux.tar.bz2",
        "995aea899b6adfb075cfe1fbebe53a33165e9e106766d979de0a9717335bab08",
    ),
}


# Thread lock to prevent race on download_llvm_files() from multi-threading.
# This works in tandem with the inter-process file lock - both are required.
_LLVM_DOWNLOAD_LOCK = Lock()
_LLVM_DOWNLOADED = False


def _download_llvm_files(unpacked_location: Path) -> Path:
    """Download and unpack the LLVM data pack."""
    global _LLVM_DOWNLOADED
    _LLVM_DOWNLOADED = True
    if not (unpacked_location / ".unpacked").is_file():
        # Tidy up an incomplete unpack.
        shutil.rmtree(unpacked_location, ignore_errors=True)

        url, sha256 = _LLVM_URLS[sys.platform]
        tar_contents = io.BytesIO(download(url, sha256=sha256))
        unpacked_location.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=tar_contents, mode="r:bz2") as tar:
            tar.extractall(unpacked_location)
        assert unpacked_location.is_dir()
        assert (unpacked_location / "LICENSE").is_file()
        # Create the marker file to indicate that the directory is unpacked
        # and ready to go.
        (unpacked_location / ".unpacked").touch()

    return unpacked_location


def download_llvm_files() -> Path:
    """Download and unpack the LLVM data pack."""
    unpacked_location = site_data_path("llvm/10.0.0")
    # Fast path for repeated calls.
    if _LLVM_DOWNLOADED:
        return unpacked_location
    # Fast path for first call. This check will be repeated inside the locked
    # region if required.
    if (unpacked_location / ".unpacked").is_file():
        return unpacked_location

    with _LLVM_DOWNLOAD_LOCK:
        with fasteners.InterProcessLock(cache_path("llvm-download.LOCK")):
            return _download_llvm_files(unpacked_location)


def clang_path() -> Path:
    """Return the path of clang."""
    return download_llvm_files() / "bin/clang"


def lli_path() -> Path:
    """Return the path of lli."""
    return download_llvm_files() / "bin/lli"


def llvm_as_path() -> Path:
    """Return the path of llvm-as."""
    return download_llvm_files() / "bin/llvm-as"


def llvm_link_path() -> Path:
    """Return the path of llvm-link."""
    return download_llvm_files() / "bin/llvm-link"


def llvm_stress_path() -> Path:
    """Return the path of llvm-stress."""
    return download_llvm_files() / "bin/llvm-stress"


def opt_path() -> Path:
    """Return the path of opt."""
    return download_llvm_files() / "bin/opt"
