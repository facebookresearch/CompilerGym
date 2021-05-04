# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/datasets:files_dataset_test."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from compiler_gym.datasets import FilesDataset
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.common"]


@pytest.fixture(scope="function")
def empty_dataset() -> FilesDataset:
    with tempfile.TemporaryDirectory() as d:
        yield FilesDataset(
            name="benchmark://test-v0",
            description="",
            license="MIT",
            dataset_root=Path(d) / "files",
            site_data_base=Path(d) / "site_data",
        )


@pytest.fixture(scope="function", params=["", "memoized-ids"])
def populated_dataset(request) -> FilesDataset:
    with tempfile.TemporaryDirectory() as d:
        df = Path(d) / "files"
        (df / "a").mkdir(parents=True)
        (df / "b").mkdir()

        (df / "e.txt").touch()
        (df / "f.txt").touch()
        (df / "g.jpg").touch()
        (df / "a" / "a.txt").touch()
        (df / "a" / "b.txt").touch()
        (df / "b" / "a.txt").touch()
        (df / "b" / "b.txt").touch()
        (df / "b" / "c.txt").touch()
        (df / "b" / "d.jpg").touch()

        yield FilesDataset(
            name="benchmark://test-v0",
            description="",
            license="MIT",
            dataset_root=Path(d) / "files",
            site_data_base=Path(d) / "site_data",
            memoize_uris=request.param == "memoized-ids",
        )


def test_dataset_is_installed(empty_dataset: FilesDataset):
    assert empty_dataset.installed


def test_empty_dataset(empty_dataset: FilesDataset):
    assert empty_dataset.size == 0
    assert list(empty_dataset.benchmark_uris()) == []
    assert list(empty_dataset.benchmarks()) == []


def test_populated_dataset(populated_dataset: FilesDataset):
    for _ in range(2):
        assert list(populated_dataset.benchmark_uris()) == [
            "benchmark://test-v0/e.txt",
            "benchmark://test-v0/f.txt",
            "benchmark://test-v0/g.jpg",
            "benchmark://test-v0/a/a.txt",
            "benchmark://test-v0/a/b.txt",
            "benchmark://test-v0/b/a.txt",
            "benchmark://test-v0/b/b.txt",
            "benchmark://test-v0/b/c.txt",
            "benchmark://test-v0/b/d.jpg",
        ]
        assert populated_dataset.size == 9


def test_populated_dataset_benchmark_lookup(populated_dataset: FilesDataset):
    bm = populated_dataset.benchmark("benchmark://test-v0/e.txt")
    assert bm.uri == "benchmark://test-v0/e.txt"
    assert bm.proto.uri == "benchmark://test-v0/e.txt"
    assert bm.proto.program.uri == f"file:///{populated_dataset.dataset_root}/e.txt"


def test_populated_dataset_first_file(populated_dataset: FilesDataset):
    bm = next(populated_dataset.benchmarks())
    assert bm.uri == "benchmark://test-v0/e.txt"
    assert bm.proto.uri == "benchmark://test-v0/e.txt"
    assert bm.proto.program.uri == f"file:///{populated_dataset.dataset_root}/e.txt"


def test_populated_dataset_benchmark_lookup_not_found(populated_dataset: FilesDataset):
    with pytest.raises(LookupError) as e_ctx:
        populated_dataset.benchmark("benchmark://test-v0/not/a/file")

    assert str(e_ctx.value).startswith(
        "Benchmark not found: benchmark://test-v0/not/a/file"
    )


def test_populated_dataset_with_file_extension_filter(populated_dataset: FilesDataset):
    populated_dataset.benchmark_file_suffix = ".jpg"
    assert list(populated_dataset.benchmark_uris()) == [
        "benchmark://test-v0/g",
        "benchmark://test-v0/b/d",
    ]
    assert populated_dataset.size == 2


def test_populated_dataset_random_benchmark(populated_dataset: FilesDataset):
    num_benchmarks = 3
    rng = np.random.default_rng(0)
    random_benchmarks = {
        b.uri
        for b in (
            populated_dataset.random_benchmark(rng) for _ in range(num_benchmarks)
        )
    }
    assert len(random_benchmarks) == num_benchmarks


if __name__ == "__main__":
    main()
