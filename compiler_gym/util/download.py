# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hashlib
import logging
from time import sleep
from typing import List, Optional, Union

import fasteners
import requests

from compiler_gym.util.filesystem import atomic_file_write
from compiler_gym.util.runfiles_path import cache_path
from compiler_gym.util.truncate import truncate


class DownloadFailed(IOError):
    """Error thrown if a download fails."""


class TooManyRequests(DownloadFailed):
    """Error thrown by HTTP 429 response."""


def _get_url_data(url: str) -> bytes:
    try:
        req = requests.get(url)
    except IOError as e:
        # Re-cast an error raised by requests library to DownloadFailed type.
        raise DownloadFailed(str(e)) from e

    try:
        if req.status_code == 429:
            raise TooManyRequests("429 Too Many Requests")
        elif req.status_code != 200:
            raise DownloadFailed(f"GET returned status code {req.status_code}: {url}")

        return req.content
    finally:
        req.close()


def _do_download_attempt(url: str, sha256: Optional[str]) -> bytes:
    logging.info("Downloading %s ...", url)
    content = _get_url_data(url)
    if sha256:
        # Validate the checksum.
        checksum = hashlib.sha256()
        checksum.update(content)
        actual_sha256 = checksum.hexdigest()
        if sha256 != actual_sha256:
            raise DownloadFailed(
                f"Checksum of download does not match:\n"
                f"Url: {url}\n"
                f"Expected: {sha256}\n"
                f"Actual:   {actual_sha256}"
            )

        # Cache the downloaded file.
        path = cache_path(f"downloads/{sha256}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with atomic_file_write(path, fileobj=True) as f:
            f.write(content)

    logging.debug(f"Downloaded {url}")
    return content


def _download(urls: List[str], sha256: Optional[str], max_retries: int) -> bytes:
    if not urls:
        raise ValueError("No URLs to download")

    # Cache hit.
    if sha256 and cache_path(f"downloads/{sha256}").is_file():
        with open(str(cache_path(f"downloads/{sha256}")), "rb") as f:
            return f.read()

    # A retry loop, and loop over all urls provided.
    last_exception = None
    wait_time = 10
    for _ in range(max(max_retries, 1)):
        for url in urls:
            try:
                return _do_download_attempt(url, sha256)
            except TooManyRequests as e:
                last_exception = e
                logging.info(
                    "Download attempt failed with Too Many Requests error. "
                    "Watiting %.1f seconds",
                    wait_time,
                )
                sleep(wait_time)
                wait_time *= 1.5
            except DownloadFailed as e:
                logging.info("Download attempt failed: %s", truncate(e))
                last_exception = e
    raise last_exception


def download(
    urls: Union[str, List[str]], sha256: Optional[str] = None, max_retries: int = 5
) -> bytes:
    """Download a file and return its contents.

    If :code:`sha256` is provided and the download succeeds, the file contents
    are cached locally in :code:`$cache_path/downloads/$sha256`. See
    :func:`compiler_gym.cache_path`.

    An inter-process lock ensures that only a single call to this function may
    execute at a time.

    :param urls: Either a single URL of the file to download, or a list of URLs
        to download.

    :param sha256: The expected sha256 checksum of the file.

    :return: The contents of the downloaded file.

    :raises IOError: If the download fails, or if the downloaded content does
        match the expected :code:`sha256` checksum.
    """
    # Convert a singular string into a list of strings.
    urls = [urls] if not isinstance(urls, list) else urls

    # Only a single process may download a file at a time. The idea here is to
    # prevent redundant downloads when multiple simultaneous processes all try
    # and download the same resource. If we don't have an ID for the resource
    # then we just lock globally to reduce NIC thrashing.
    if sha256:
        with fasteners.InterProcessLock(cache_path(f"downloads/.{sha256}.lock")):
            return _download(urls, sha256, max_retries)
    else:
        with fasteners.InterProcessLock(cache_path("downloads/.lock")):
            return _download(urls, None, max_retries)
