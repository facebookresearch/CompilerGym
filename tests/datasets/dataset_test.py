# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/datasets."""
import re
from pathlib import Path

import pytest

from compiler_gym.datasets.dataset import BENCHMARK_URI_RE, DATASET_NAME_RE, Dataset
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.common"]


def _rgx_match(regex, groupname, string):
    match = re.match(regex, string)
    assert match, f"Failed to match regex '{regex}' using string '{groupname}'"
    return match.group(groupname)


@pytest.mark.parametrize("regex", (DATASET_NAME_RE, BENCHMARK_URI_RE))
def test_benchmark_uri_protocol(regex):
    assert not regex.match("B?://cbench-v1/")  # Invalid characters
    assert not regex.match("cbench-v1/")  # Missing protocol

    _rgx_match(regex, "dataset_protocol", "benchmark://cbench-v1/") == "benchmark"
    _rgx_match(regex, "dataset_protocol", "Generator13://gen-v11/") == "Generator13"


def test_benchmark_uri_dataset():
    assert not BENCHMARK_URI_RE.match("benchmark://cbench-v1")  # Missing trailing /
    assert not BENCHMARK_URI_RE.match("benchmark://cBench?v0/")  # Invalid character
    assert not BENCHMARK_URI_RE.match("benchmark://cBench/")  # Missing version suffix

    _rgx_match(
        BENCHMARK_URI_RE, "dataset_name", "benchmark://cbench-v1/"
    ) == "cbench-v1"
    _rgx_match(BENCHMARK_URI_RE, "dataset_name", "Generator13://gen-v11/") == "gen-v11"


def test_benchmark_dataset_name():
    _rgx_match(
        BENCHMARK_URI_RE, "dataset", "benchmark://cbench-v1/"
    ) == "benchmark://cbench-v1"
    _rgx_match(
        BENCHMARK_URI_RE, "dataset", "Generator13://gen-v11/"
    ) == "Generator13://gen-v11"


def test_benchmark_uri_id():
    assert not BENCHMARK_URI_RE.match("benchmark://cbench-v1/ whitespace")  # Whitespace
    assert not BENCHMARK_URI_RE.match("benchmark://cbench-v1/\t")  # Whitespace

    _rgx_match(BENCHMARK_URI_RE, "benchmark_name", "benchmark://cbench-v1/") == ""
    _rgx_match(BENCHMARK_URI_RE, "benchmark_name", "benchmark://cbench-v1/foo") == "foo"
    _rgx_match(
        BENCHMARK_URI_RE, "benchmark_name", "benchmark://cbench-v1/foo/123"
    ) == "foo/123"
    _rgx_match(
        BENCHMARK_URI_RE,
        "benchmark_name",
        "benchmark://cbench-v1/foo/123?param=true&false",
    ) == "foo/123?param=true&false"


def test_dataset_properties(tmpwd: Path):
    class TestDataset(Dataset):
        def __init__(self):
            super().__init__(
                name="benchmark://test-v0",
                description="A test dataset",
                license="MIT",
                site_data_base="test",
            )

    dataset = TestDataset()
    assert dataset.name == "benchmark://test-v0"
    assert dataset.protocol == "benchmark"
    assert dataset.version == 0
    assert dataset.description == "A test dataset"
    assert dataset.license == "MIT"
    assert dataset.long_description_url is None


def test_dataset_site_data_directory(tmpwd: Path):
    class TestDataset(Dataset):
        def __init__(self):
            super().__init__(
                name="benchmark://test-v0",
                description="A test dataset",
                license="MIT",
                site_data_base="test",
            )

    dataset = TestDataset()
    # Use endswith() since tmpwd on macOS may have /private prefix.
    assert str(dataset.site_data_path).endswith(
        str(tmpwd / "test" / "benchmark" / "test-v0")
    )
    assert not dataset.site_data_path.is_dir()  # Dir is not created until needed.


def test_dataset_long_description_url(tmpwd: Path):
    class TestDataset(Dataset):
        def __init__(self):
            super().__init__(
                name="benchmark://test-v0",
                description="A test dataset",
                license="MIT",
                long_description_url="https://facebook.com",
                site_data_base="test",
            )

    dataset = TestDataset()
    assert dataset.long_description_url == "https://facebook.com"


def test_dataset_name_missing_version(tmpwd: Path):
    class TestDataset(Dataset):
        def __init__(self):
            super().__init__(
                name="benchmark://test",
                description="A test dataset",
                license="MIT",
                site_data_base="test",
            )

    with pytest.raises(ValueError) as e_ctx:
        TestDataset()

    assert "Invalid dataset name: 'benchmark://test'" in str(e_ctx.value)


if __name__ == "__main__":
    main()
