# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/datasets."""
from pathlib import Path

import pytest

from compiler_gym.datasets.dataset import Dataset
from compiler_gym.datasets.uri import BenchmarkUri
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.common"]

# pylint: disable=abstract-method


def test_dataset_properties():
    """Test the dataset property values."""
    dataset = Dataset(
        name="benchmark://test-v0",
        description="A test dataset",
        license="MIT",
        site_data_base="test",
    )

    assert dataset.name == "benchmark://test-v0"
    assert dataset.scheme == "benchmark"
    assert dataset.description == "A test dataset"
    assert dataset.license == "MIT"


def test_dataset_optional_properties():
    """Test the default values of optional dataset properties."""
    dataset = Dataset(
        name="benchmark://test-v0",
        description="A test dataset",
        license="MIT",
        site_data_base="test",
    )

    assert dataset.references == {}  # Default value.
    assert not dataset.deprecated
    assert dataset.sort_order == 0
    assert dataset.validatable == "No"


def test_dataset_default_version():
    """Test the dataset property values."""
    dataset = Dataset(
        name="benchmark://test",
        description="A test dataset",
        license="MIT",
        site_data_base="test",
    )

    assert dataset.name == "benchmark://test"
    assert dataset.scheme == "benchmark"
    assert dataset.version == 0


def test_dataset_optional_properties_explicit_values():
    """Test the non-default values of optional dataset properties."""
    dataset = Dataset(
        name="benchmark://test-v0",
        description="A test dataset",
        license="MIT",
        site_data_base="test",
        references={"GitHub": "https://github.com/facebookresearch/CompilerGym"},
        deprecated="Deprecation message",
        sort_order=10,
        validatable="Yes",
    )

    assert dataset.references == {
        "GitHub": "https://github.com/facebookresearch/CompilerGym"
    }
    assert dataset.deprecated
    assert dataset.sort_order == 10
    assert dataset.validatable == "Yes"


def test_dataset_inferred_properties():
    """Test the values of inferred dataset properties."""
    dataset = Dataset(
        name="benchmark://test-v2",
        description="A test dataset",
        license="MIT",
        site_data_base="test",
    )

    assert dataset.scheme == "benchmark"
    assert dataset.version == 2


def test_dataset_properties_read_only(tmpwd: Path):
    """Test that dataset properties are read-only."""
    dataset = Dataset(
        name="benchmark://test-v0",
        description="A test dataset",
        license="MIT",
        site_data_base="test",
    )

    with pytest.raises(AttributeError):
        dataset.name = "benchmark://test-v1"
    with pytest.raises(AttributeError):
        dataset.description = "A test dataset"
    with pytest.raises(AttributeError):
        dataset.license = "MIT"
    with pytest.raises(AttributeError):
        dataset.site_data_path = tmpwd


def test_dataset_site_data_directory(tmpwd: Path):
    """Test the path generated for site data."""
    dataset = Dataset(
        name="benchmark://test-v0",
        description="A test dataset",
        license="MIT",
        site_data_base="test",
    )

    # Use endswith() since tmpwd on macOS may have a '/private' prefix.
    assert str(dataset.site_data_path).endswith(
        str(tmpwd / "test" / "benchmark" / "test-v0")
    )
    assert not dataset.site_data_path.is_dir()  # Dir is not created until needed.


def test_dataset_deprecation_message(tmpwd: Path):
    """Test that a deprecation warning is emitted on install()."""
    dataset = Dataset(
        name="benchmark://test-v0",
        description="A test dataset",
        license="MIT",
        site_data_base="test",
        deprecated="The cat sat on the mat",
    )

    with pytest.warns(DeprecationWarning, match="The cat sat on the mat"):
        dataset.install()


def test_dataset_equality_and_sorting():
    """Test comparison operators between datasets."""
    a = Dataset(
        name="benchmark://a-v0",
        description="A test dataset",
        license="MIT",
        site_data_base="test",
    )
    a2 = Dataset(
        name="benchmark://a-v0",
        description="A test dataset",
        license="MIT",
        site_data_base="test",
    )
    b = Dataset(
        name="benchmark://b-v0",
        description="A test dataset",
        license="MIT",
        site_data_base="test",
    )
    assert a == a2
    assert a != b
    assert a < b
    assert a <= b
    assert b > a
    assert b >= a

    # String comparisons
    assert a == "benchmark://a-v0"
    assert a != "benchmark://b-v0"
    assert a < "benchmark://b-v0"

    # Sorting
    assert sorted([a2, b, a]) == [
        "benchmark://a-v0",
        "benchmark://a-v0",
        "benchmark://b-v0",
    ]


class DatasetForTesting(Dataset):
    """A dataset to use for testing."""

    def __init__(self, benchmarks=None):
        super().__init__(
            name="benchmark://test-v0",
            description="A test dataset",
            license="MIT",
            site_data_base="test",
        )
        self._benchmarks = benchmarks or {
            "benchmark://test-v0/a": 1,
            "benchmark://test-v0/b": 2,
            "benchmark://test-v0/c": 3,
        }

    def benchmark_uris(self):
        return sorted(self._benchmarks)

    def benchmark_from_parsed_uri(self, uri: BenchmarkUri):
        return self._benchmarks[str(uri)]

    @property
    def size(self):
        return len(self._benchmarks)


def test_dataset_size():
    dataset = DatasetForTesting()
    assert dataset.size == 3
    assert len(dataset) == 3


def test_benchmarks_lookup_by_uri():
    dataset = DatasetForTesting()
    assert dataset.benchmark("benchmark://test-v0/b") == 2
    assert dataset["benchmark://test-v0/b"] == 2


def test_benchmarks_iter():
    dataset = DatasetForTesting()
    assert list(dataset.benchmarks()) == [1, 2, 3]
    assert list(dataset) == [1, 2, 3]


def test_logger_is_deprecated():
    dataset = DatasetForTesting()
    with pytest.deprecated_call(match="The `Dataset.logger` attribute is deprecated"):
        dataset.logger


def test_with_site_data():
    """Test the dataset property values."""
    dataset = Dataset(
        name="benchmark://test-v0",
        description="A test dataset",
        license="MIT",
        site_data_base="test",
    )
    assert dataset.has_site_data


def test_without_site_data():
    """Test the dataset property values."""
    dataset = Dataset(
        name="benchmark://test-v0",
        description="A test dataset",
        license="MIT",
    )
    assert not dataset.has_site_data
    with pytest.raises(
        ValueError, match=r"^Dataset has no site data path: benchmark://test-v0$"
    ):
        dataset.site_data_path  # noqa


if __name__ == "__main__":
    main()
