# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hashlib
import logging
from typing import Optional

import fasteners
import requests

from compiler_gym.util.runfiles_path import cache_path


def _download(url: str) -> bytes:
    req = requests.get(url)
    try:
        if req.status_code != 200:
            raise OSError(f"GET returned status code {req.status_code}: {url}")

        logging.info(f"Downloaded {url}")
        return req.content
    finally:
        req.close()


# Only a single process may download at a time. The idea here is to prevent
# overloading the NIC when, for example, you launch a bunch of simultaneous
# learning processes which all require the same dataset.
@fasteners.interprocess_locked(cache_path(f"downloads/LOCK"))
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
    # Cache hit.
    if sha256 and cache_path(f"downloads/{sha256}").is_file():
        with open(str(cache_path(f"downloads/{sha256}")), "rb") as f:
            return f.read()

    logging.info(f"Downloading {url} ...")
    content = _download(url)
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
        with open(str(cache_path(f"downloads/{sha256}")), "wb") as f:
            f.write(content)

    logging.info(f"Downloaded {url}")
    return content
