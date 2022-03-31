# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Module for resolving paths to LLVM binaries and libraries."""
import io
import logging
import os
import shutil
import sys
import tarfile
from pathlib import Path
from threading import Lock
from typing import Optional

from fasteners import InterProcessLock

from compiler_gym.util.download import download
from compiler_gym.util.runfiles_path import cache_path, site_data_path

logger = logging.getLogger(__name__)

# The data archive containing LLVM binaries and libraries.
# HACK: I had to use the original llvm download binaries so that they contain the include files I need
_LLVM_URL, _LLVM_SHA256 = {
    "darwin": (
        "https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.0/clang+llvm-10.0.0-x86_64-apple-darwin.tar.xz",  # "https://dl.fbaipublicfiles.com/compiler_gym/llvm-v0-macos.tar.bz2",
        "633a833396bf2276094c126b072d52b59aca6249e7ce8eae14c728016edb5e61",  # "731ae351b62c5713fb5043e0ccc56bfba4609e284dc816f0b2a5598fb809bf6b",
    ),
    "linux": (
        "https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.0/clang+llvm-10.0.0-aarch64-linux-gnu.tar.xz",  # "https://dl.fbaipublicfiles.com/compiler_gym/llvm-v0-linux.tar.bz2",
        "c2072390dc6c8b4cc67737f487ef384148253a6a97b38030e012c4d7214b7295",  # "59c3f328efd51994a11168ca15e43a8d422233796c6bc167c9eb771c7bd6b57e",
    ),
}[sys.platform]


# Thread lock to prevent race on download_llvm_files() from multi-threading.
# This works in tandem with the inter-process file lock - both are required.
_LLVM_DOWNLOAD_LOCK = Lock()
_LLVM_UNPACKED_LOCATION: Optional[Path] = None


def _download_llvm_files(destination: Path) -> Path:
    """Download and unpack the LLVM data pack."""
    logger.warning(
        "Installing the CompilerGym LLVM environment runtime. This may take a few moments ..."
    )

    # Tidy up an incomplete unpack.
    shutil.rmtree(destination, ignore_errors=True)

    tar_contents = io.BytesIO(download(_LLVM_URL, sha256=_LLVM_SHA256))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=tar_contents, mode="r:xz") as tar:
        tar.extractall(destination)
    # HACK: llvm original tar files have a root directory which we want to bypass
    sub_dir = os.listdir(destination)[0]
    for item in os.listdir(os.path.join(str(destination), sub_dir)):
        shutil.move(
            os.path.join(str(destination), sub_dir, item),
            os.path.join(str(destination), item),
        )
    shutil.rmtree(os.path.join(str(destination), sub_dir), ignore_errors=True)
    # HACK: the downloaded binaries from LLVM website has "clang-10" rather than "clang"
    if os.path.exists(os.path.join(str(destination), "clang-10")) and not os.path.join(
        str(destination), "clang"
    ):
        shutil.move(
            os.path.join(str(destination), "clang-10"),
            os.path.join(str(destination), "clang"),
        )

    assert destination.is_dir()
    # HACK: llvm original downloads don't contain a LIENSE file
    # assert (destination / "LICENSE").is_file()

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


def llc_path() -> Path:
    """Return the path of llc."""
    return download_llvm_files() / "bin/llc"


def llvm_as_path() -> Path:
    """Return the path of llvm-as."""
    return download_llvm_files() / "bin/llvm-as"


def llvm_dis_path() -> Path:
    """Return the path of llvm-as."""
    return download_llvm_files() / "bin/llvm-dis"


def llvm_link_path() -> Path:
    """Return the path of llvm-link."""
    return download_llvm_files() / "bin/llvm-link"


def llvm_stress_path() -> Path:
    """Return the path of llvm-stress."""
    return download_llvm_files() / "bin/llvm-stress"


def llvm_diff_path() -> Path:
    """Return the path of llvm-diff."""
    return download_llvm_files() / "bin/llvm-diff"


def opt_path() -> Path:
    """Return the path of opt."""
    return download_llvm_files() / "bin/opt"
