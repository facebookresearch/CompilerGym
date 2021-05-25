# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/util:download."""
import pytest

from compiler_gym.util import download
from compiler_gym.util.runfiles_path import cache_path
from tests.test_main import main


@pytest.mark.parametrize("max_retries", [1, 2, 3, 5, 10])
def test_download_timeout_retry_loop(mocker, max_retries: int):
    """Check that download attempts are repeated with sleep() on error."""

    def patched_download(*args):
        raise download.TooManyRequests

    mocker.patch.object(download, "sleep")
    mocker.patch.object(download, "_do_download_attempt", patched_download)
    mocker.spy(download, "_do_download_attempt")

    with pytest.raises(download.TooManyRequests):
        download.download(urls="example", max_retries=max_retries)

    assert download._do_download_attempt.call_count == max_retries
    assert download.sleep.call_count == max_retries
    download.sleep.assert_called_with(5 * 1.5 ** (max_retries - 1))


@pytest.mark.parametrize("max_retries", [1, 2, 3, 5, 10])
def test_download_failed_retry_loop(mocker, max_retries: int):
    """Check that download attempts are repeated without sleep() on error."""

    def patched_download(*args):
        raise download.DownloadFailed

    mocker.patch.object(download, "sleep")
    mocker.patch.object(download, "_do_download_attempt", patched_download)
    mocker.spy(download, "_do_download_attempt")

    with pytest.raises(download.DownloadFailed):
        download.download(urls="example", max_retries=max_retries)

    assert download._do_download_attempt.call_count == max_retries
    assert download.sleep.call_count == 0


def test_download_cache_hit(mocker):
    """Check that download is not repeated on cache hit."""
    data = b"Hello, world"
    data_checksum = "4ae7c3b6ac0beff671efa8cf57386151c06e58ca53a78d83f36107316cec125f"
    cached_path = cache_path(f"downloads/{data_checksum}")

    # Tidy up from a previous test, if applicable.
    if cached_path.is_file():
        cached_path.unlink()

    def patched_download(*args):
        return data

    mocker.patch.object(download, "_get_url_data", patched_download)
    mocker.spy(download, "_get_url_data")

    assert (
        download.download(
            "example",
            sha256="4ae7c3b6ac0beff671efa8cf57386151c06e58ca53a78d83f36107316cec125f",
        )
        == data
    )
    download._get_url_data.assert_called_once_with("example")
    assert cached_path.is_file()

    # Cache hit.
    assert (
        download.download(
            "example",
            sha256="4ae7c3b6ac0beff671efa8cf57386151c06e58ca53a78d83f36107316cec125f",
        )
        == data
    )
    assert download._get_url_data.call_count == 1


def test_download_mismatched_checksum(mocker):
    """Check that error is raised when checksum does not match expected."""

    def patched_download(*args):
        return b"Hello, world"

    mocker.patch.object(download, "_get_url_data", patched_download)

    with pytest.raises(
        download.DownloadFailed, match="Checksum of download does not match"
    ):
        download.download("example", sha256="123")


def test_download_no_urls():
    with pytest.raises(ValueError, match="No URLs to download"):
        download.download(urls=[])


if __name__ == "__main__":
    main()
