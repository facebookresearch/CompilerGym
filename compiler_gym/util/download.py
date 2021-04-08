# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hashlib
import logging
import os
from typing import Optional

import fasteners
import requests

from compiler_gym.util.runfiles_path import cache_path


def __download(url: str) -> bytes:
    req = requests.get(url)
    try:
        if req.status_code != 200:
            raise OSError(f"GET returned status code {req.status_code}: {url}")

        return req.content
    finally:
        req.close()


def _download(url: str, sha256: Optional[str]) -> bytes:
    # Cache hit.
    if sha256 and cache_path(f"downloads/{sha256}").is_file():
        with open(str(cache_path(f"downloads/{sha256}")), "rb") as f:
            return f.read()

    logging.info(f"Downloading {url} ...")
    content = __download(url)
    if sha256:
        # Validate the checksum.
        checksum = hashlib.sha256()
        checksum.update(content)
        actual_sha256 = checksum.hexdigest()
        if sha256 != actual_sha256:
            raise OSError(
                f"Checksum of downloaded dataset does not match:\n"
                f"Url: {url}\n"
                f"Expected: {sha256}\n"
                f"Actual:   {actual_sha256}"
            )

        # Cache the downloaded file.
        cache_path("downloads").mkdir(parents=True, exist_ok=True)
        # Atomic write by writing to a temporary file and renaming.
        manifest_path = cache_path(f"downloads/{sha256}")
        with open(f"{manifest_path}.tmp", "wb") as f:
            f.write(content)
        os.rename(f"{manifest_path}.tmp", str(manifest_path))

    logging.debug(f"Downloaded {url}")
    return content


@fasteners.interprocess_locked(cache_path("downloads/LOCK"))
def download(url: str, sha256: Optional[str] = None) -> bytes:
    """Download a file and return its contents.

    If :code:`sha256` is provided and the download succeeds, the file contents are cached locally
    in :code:`$cache_path/downloads/$sha256`. See :func:`compiler_gym.cache_path`.

    An inter-process lock ensures that only a single call to this function may
    execute at a time.

    :param url: The URL of the file to download.
    :param sha256: The expected sha256 checksum of the file.
    :return: The contents of the downloaded file.
    :raises OSError: If the download fails, or if the downloaded content does match the expected
        :code:`sha256` checksum.
    """
    # Only a single process may download a file at a time. The idea here is to
    # prevent redundant downloads when multiple simultaneous processes all try
    # and download the same resource. If we don't have an ID for the resource
    # then we just lock globally to reduce NIC thrashing.
    if sha256:
        with fasteners.InterProcessLock(cache_path("downloads/{sha256}.lock")):
            return _download(url, sha256)
    else:
        with fasteners.InterProcessLock(cache_path("downloads/LOCK")):
            return _download(url, None)
