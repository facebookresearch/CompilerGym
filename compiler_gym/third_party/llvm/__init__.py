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
from typing import Optional

from fasteners import InterProcessLock

from compiler_gym.util.download import download
from compiler_gym.util.runfiles_path import cache_path, site_data_path

# The data archive containing LLVM binaries and libraries.
_LLVM_URL, _LLVM_SHA256 = {
    "darwin": (
        "https://dl.fbaipublicfiles.com/compiler_gym/llvm-v0-macos.tar.bz2",
        "731ae351b62c5713fb5043e0ccc56bfba4609e284dc816f0b2a5598fb809bf6b",
    ),
    "linux": (
        "https://dl.fbaipublicfiles.com/compiler_gym/llvm-v0-linux.tar.bz2",
        "59c3f328efd51994a11168ca15e43a8d422233796c6bc167c9eb771c7bd6b57e",
    ),
}[sys.platform]


# Thread lock to prevent race on download_llvm_files() from multi-threading.
# This works in tandem with the inter-process file lock - both are required.
_LLVM_DOWNLOAD_LOCK = Lock()
_LLVM_UNPACKED_LOCATION: Optional[Path] = None


def _download_llvm_files(destination: Path) -> Path:
    """Download and unpack the LLVM data pack."""
    # Tidy up an incomplete unpack.
    shutil.rmtree(destination, ignore_errors=True)

    tar_contents = io.BytesIO(download(_LLVM_URL, sha256=_LLVM_SHA256))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=tar_contents, mode="r:bz2") as tar:
        tar.extractall(destination)

    assert destination.is_dir()
    assert (destination / "LICENSE").is_file()

    return destination


def download_llvm_files() -> Path:
    """Download and unpack the LLVM data pack."""
    global _LLVM_UNPACKED_LOCATION

    unpacked_location = site_data_path("llvm-v0")
    # Fast path for repeated calls.
    if _LLVM_UNPACKED_LOCATION == unpacked_location:
        return unpacked_location

    with _LLVM_DOWNLOAD_LOCK:
        # Fast path for first call. This check will be repeated inside the locked
        # region if required.
        if (unpacked_location / ".unpacked").is_file():
            _LLVM_UNPACKED_LOCATION = unpacked_location
            return unpacked_location

        with InterProcessLock(cache_path(".llvm-v0-install.LOCK")):
            # Now that the lock is acquired, repeat the check to see if it is
            # necessary to download the dataset.
            if (unpacked_location / ".unpacked").is_file():
                return unpacked_location

            _download_llvm_files(unpacked_location)
            # Create the marker file to indicate that the directory is unpacked
            # and ready to go.
            (unpacked_location / ".unpacked").touch()
            _LLVM_UNPACKED_LOCATION = unpacked_location

        return unpacked_location


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
