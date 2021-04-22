# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/datasets:files_dataset_test."""
import tempfile
from pathlib import Path
from typing import Optional

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
    assert empty_dataset.n == 0
    assert list(empty_dataset.benchmark_uris()) == []
    assert list(empty_dataset.benchmarks()) == []


def test_empty_dataset_benchmark(empty_dataset: FilesDataset):
    with pytest.raises(ValueError) as e_ctx:
        empty_dataset.benchmark()

    assert str(e_ctx.value) == "No benchmarks"


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
        assert populated_dataset.n == 9


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
    assert populated_dataset.n == 2


@pytest.mark.parametrize(
    "requested_uri", (None, "benchmark://test-v0", "benchmark://test-v0/")
)
def test_populated_dataset_random_benchmark(
    populated_dataset: FilesDataset, requested_uri: Optional[str]
):
    populated_dataset.benchmark_file_suffix = ".jpg"

    populated_dataset.seed(1)
    benchmarks_a = [populated_dataset.benchmark(requested_uri).uri for _ in range(50)]
    populated_dataset.seed(1)
    benchmarks_b = [populated_dataset.benchmark(requested_uri).uri for _ in range(50)]
    populated_dataset.seed(2)
    benchmarks_c = [populated_dataset.benchmark(requested_uri).uri for _ in range(50)]

    assert benchmarks_a == benchmarks_b
    assert benchmarks_b != benchmarks_c

    assert set(benchmarks_a) == set(populated_dataset.benchmark_uris())
    assert set(benchmarks_c) == set(populated_dataset.benchmark_uris())


def test_populated_dataset_get_benchmark_by_index(populated_dataset: FilesDataset):
    # pylint: disable=protected-access

    i = 0
    benchmark = populated_dataset._get_benchmark_by_index(i)
    assert benchmark.uri == "benchmark://test-v0/e.txt"
    assert Path(benchmark.proto.program.uri[len("file:///") :]).is_file()

    i += 1
    benchmark = populated_dataset._get_benchmark_by_index(i)
    assert benchmark.uri == "benchmark://test-v0/f.txt"
    assert Path(benchmark.proto.program.uri[len("file:///") :]).is_file()

    i += 1
    benchmark = populated_dataset._get_benchmark_by_index(i)
    assert benchmark.uri == "benchmark://test-v0/g.jpg"
    assert Path(benchmark.proto.program.uri[len("file:///") :]).is_file()

    i += 1
    benchmark = populated_dataset._get_benchmark_by_index(i)
    assert benchmark.uri == "benchmark://test-v0/a/a.txt"
    assert Path(benchmark.proto.program.uri[len("file:///") :]).is_file()

    i += 1
    benchmark = populated_dataset._get_benchmark_by_index(i)
    assert benchmark.uri == "benchmark://test-v0/a/b.txt"
    assert Path(benchmark.proto.program.uri[len("file:///") :]).is_file()

    i += 1
    benchmark = populated_dataset._get_benchmark_by_index(i)
    assert benchmark.uri == "benchmark://test-v0/b/a.txt"
    assert Path(benchmark.proto.program.uri[len("file:///") :]).is_file()

    i += 1
    benchmark = populated_dataset._get_benchmark_by_index(i)
    assert benchmark.uri == "benchmark://test-v0/b/b.txt"
    assert Path(benchmark.proto.program.uri[len("file:///") :]).is_file()

    i += 1
    benchmark = populated_dataset._get_benchmark_by_index(i)
    assert benchmark.uri == "benchmark://test-v0/b/c.txt"
    assert Path(benchmark.proto.program.uri[len("file:///") :]).is_file()

    i += 1
    benchmark = populated_dataset._get_benchmark_by_index(i)
    assert benchmark.uri == "benchmark://test-v0/b/d.jpg"
    assert Path(benchmark.proto.program.uri[len("file:///") :]).is_file()


def test_populated_dataset_get_benchmark_by_index_out_of_range(
    populated_dataset: FilesDataset,
):
    # pylint: disable=protected-access
    with pytest.raises(IndexError):
        populated_dataset._get_benchmark_by_index(-1)
        populated_dataset._get_benchmark_by_index(10)


if __name__ == "__main__":
    main()
