# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/datasets."""
from pathlib import Path

import numpy as np
import pytest

from compiler_gym.datasets.datasets import Datasets, round_robin_iterables
from compiler_gym.datasets.uri import BenchmarkUri
from compiler_gym.service.proto import Benchmark as BenchmarkProto
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.common"]


class MockDataset:
    """A mock Dataset class."""

    def __init__(self, name):
        self.name = name
        self.installed = False
        self.deprecated = False
        self.benchmark_values = []
        self.sort_order = 0

    def install(self):
        self.installed = True

    def uninstall(self):
        self.installed = False

    def benchmark_uris(self):
        return (b.uri for b in self.benchmark_values)

    def benchmarks(self):
        yield from self.benchmark_values

    def benchmark_from_parsed_uri(self, uri: BenchmarkUri):
        for b in self.benchmark_values:
            if b.uri == str(uri):
                return b
        raise KeyError(str(uri))

    def random_benchmark(self, random_state=None):
        return random_state.choice(self.benchmark_values)

    def __repr__(self):
        return str(self.name)


class MockBenchmark:
    """A mock Benchmark class."""

    def __init__(self, uri):
        self.uri = uri

    def __repr__(self):
        return str(self.uri)


def test_enumerate_datasets_empty():
    datasets = Datasets([])

    assert list(datasets) == []


def test_enumerate_datasets():
    da = MockDataset("benchmark://a")
    db = MockDataset("benchmark://b")
    datasets = Datasets((da, db))

    assert list(datasets) == [da, db]


def test_enumerate_datasets_with_custom_sort_order():
    da = MockDataset("benchmark://a")
    db = MockDataset("benchmark://b")
    db.sort_order = -1
    datasets = Datasets((da, db))

    assert list(datasets) == [db, da]


def test_enumerate_deprecated_datasets():
    da = MockDataset("benchmark://a")
    db = MockDataset("benchmark://b")
    datasets = Datasets((da, db))

    db.deprecated = True
    assert list(datasets) == [da]
    assert list(datasets.datasets(with_deprecated=True)) == [da, db]


def test_enumerate_datasets_deprecated_at_construction_time():
    da = MockDataset("benchmark://a")
    db = MockDataset("benchmark://b")
    db.deprecated = True
    datasets = Datasets((da, db))

    assert list(datasets) == [da]
    assert list(datasets.datasets(with_deprecated=True)) == [da, db]


def test_datasets_add_dataset():
    datasets = Datasets([])

    da = MockDataset("benchmark://foo-v0")
    datasets["benchmark://foo-v0"] = da

    assert list(datasets) == [da]


def test_datasets_add_deprecated_dataset():
    datasets = Datasets([])

    da = MockDataset("benchmark://a")
    da.deprecated = True
    datasets["benchmark://foo-v0"] = da

    assert list(datasets) == []


def test_datasets_remove():
    da = MockDataset("benchmark://foo-v0")
    datasets = Datasets([da])

    del datasets["benchmark://foo-v0"]
    assert list(datasets) == []


def test_datasets_get_item():
    da = MockDataset("benchmark://foo-v0")
    datasets = Datasets([da])

    assert datasets.dataset("benchmark://foo-v0") == da
    assert datasets["benchmark://foo-v0"] == da


def test_datasets_get_item_default_scheme():
    da = MockDataset("benchmark://foo-v0")
    datasets = Datasets([da])

    assert datasets.dataset("foo-v0") == da
    assert datasets["foo-v0"] == da


def test_datasets_get_item_lookup_miss():
    da = MockDataset("benchmark://foo-v0")
    datasets = Datasets([da])

    with pytest.raises(LookupError, match=r"^Dataset not found: benchmark://bar-v0$"):
        datasets.dataset("benchmark://bar-v0")

    with pytest.raises(LookupError, match=r"^Dataset not found: benchmark://bar-v0$"):
        _ = datasets["benchmark://bar-v0"]


def test_benchmark_lookup_by_uri():
    da = MockDataset("benchmark://foo-v0")
    db = MockDataset("benchmark://bar-v0")
    ba = MockBenchmark(uri="benchmark://foo-v0/abc")
    da.benchmark_values.append(ba)
    datasets = Datasets([da, db])

    assert datasets.benchmark("benchmark://foo-v0/abc") == ba


def test_round_robin():
    iters = iter(
        [
            iter([0, 1, 2, 3, 4, 5]),
            iter(["a", "b", "c"]),
            iter([0.5, 1.0]),
        ]
    )
    assert list(round_robin_iterables(iters)) == [
        0,
        "a",
        0.5,
        1,
        "b",
        1.0,
        2,
        "c",
        3,
        4,
        5,
    ]


def test_benchmark_uris_order():
    da = MockDataset("benchmark://foo-v0")
    db = MockDataset("benchmark://bar-v0")
    ba = MockBenchmark(uri="benchmark://foo-v0/abc")
    bb = MockBenchmark(uri="benchmark://foo-v0/123")
    bc = MockBenchmark(uri="benchmark://bar-v0/abc")
    bd = MockBenchmark(uri="benchmark://bar-v0/123")
    da.benchmark_values.append(ba)
    da.benchmark_values.append(bb)
    db.benchmark_values.append(bc)
    db.benchmark_values.append(bd)
    datasets = Datasets([da, db])

    assert list(datasets.benchmark_uris()) == [b.uri for b in datasets.benchmarks()]
    # Datasets are ordered by name, so bar-v0 before foo-v0.
    assert list(datasets.benchmark_uris()) == [
        "benchmark://bar-v0/abc",
        "benchmark://foo-v0/abc",
        "benchmark://bar-v0/123",
        "benchmark://foo-v0/123",
    ]


def test_benchmarks_iter_deprecated():
    da = MockDataset("benchmark://foo-v0")
    db = MockDataset("benchmark://bar-v0")
    db.deprecated = True
    ba = MockBenchmark(uri="benchmark://foo-v0/abc")
    bb = MockBenchmark(uri="benchmark://foo-v0/123")
    bc = MockBenchmark(uri="benchmark://bar-v0/abc")
    bd = MockBenchmark(uri="benchmark://bar-v0/123")
    da.benchmark_values.append(ba)
    da.benchmark_values.append(bb)
    db.benchmark_values.append(bc)
    db.benchmark_values.append(bd)
    datasets = Datasets([da, db])

    # Iterate over the benchmarks. The deprecated dataset is not included.
    assert list(datasets.benchmark_uris()) == [b.uri for b in datasets.benchmarks()]
    assert list(datasets.benchmark_uris()) == [
        "benchmark://foo-v0/abc",
        "benchmark://foo-v0/123",
    ]

    # Repeat the above, but include the deprecated datasets.
    assert list(datasets.benchmark_uris(with_deprecated=True)) == [
        b.uri for b in datasets.benchmarks(with_deprecated=True)
    ]
    assert list(datasets.benchmark_uris(with_deprecated=True)) == [
        "benchmark://bar-v0/abc",
        "benchmark://foo-v0/abc",
        "benchmark://bar-v0/123",
        "benchmark://foo-v0/123",
    ]


@pytest.mark.parametrize("weighted", [False, True])
def test_random_benchmark(mocker, weighted: bool):
    da = MockDataset("benchmark://foo-v0")
    da.size = 10
    ba = MockBenchmark(uri="benchmark://foo-v0/abc")
    da.benchmark_values.append(ba)
    datasets = Datasets([da])

    mocker.spy(da, "random_benchmark")

    num_benchmarks = 5
    rng = np.random.default_rng(0)
    random_benchmarks = {
        b.uri
        for b in (
            datasets.random_benchmark(rng, weighted=weighted)
            for _ in range(num_benchmarks)
        )
    }

    assert da.random_benchmark.call_count == num_benchmarks
    assert len(random_benchmarks) == 1
    assert next(iter(random_benchmarks)) == "benchmark://foo-v0/abc"


def test_dataset_proto_scheme(tmpdir):
    """Test the proto:// scheme handler."""
    tmpdir = Path(tmpdir)
    datasets = Datasets(datasets={})

    proto = BenchmarkProto(uri="hello world")
    with open(tmpdir / "file.pb", "wb") as f:
        f.write(proto.SerializeToString())

    benchmark = datasets.benchmark(f"proto://{tmpdir}/file.pb")

    assert benchmark.proto.uri == "hello world"
    assert benchmark.uri == "benchmark://hello world"


def test_dataset_proto_scheme_file_not_found(tmpdir):
    tmpdir = Path(tmpdir)
    datasets = Datasets(datasets={})
    with pytest.raises(FileNotFoundError):
        datasets.benchmark(f"proto://{tmpdir}/not_a_file")


def test_dataset_file_scheme(tmpdir):
    """Test the file:// scheme handler."""
    tmpdir = Path(tmpdir)
    datasets = Datasets(datasets={})

    with open(tmpdir / "file.dat", "w") as f:
        f.write("hello, world")

    benchmark = datasets.benchmark(f"file://{tmpdir}/file.dat")

    assert benchmark.proto.uri == f"file://{tmpdir}/file.dat"
    assert benchmark.proto.program.contents == b"hello, world"
    assert benchmark.uri == f"file://{tmpdir}/file.dat"


if __name__ == "__main__":
    main()
