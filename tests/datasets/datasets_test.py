# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/datasets."""
import pytest

from compiler_gym.datasets.datasets import Datasets, round_robin_iterables
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.common"]


class MockDataset:
    """A mock Dataset class."""

    def __init__(self, name):
        self.name = name
        self.installed = False
        self.seed_value = None
        self.hidden = False
        self.benchmark_values = []
        self.sort_order = 0

    def install(self):
        self.installed = True

    def uninstall(self):
        self.installed = False

    def seed(self, seed):
        self.seed_value = seed

    def benchmark_uris(self):
        return (b.uri for b in self.benchmark_values)

    def benchmarks(self):
        yield from self.benchmark_values

    def benchmark(self, uri=None):
        if uri:
            for b in self.benchmark_values:
                if b.uri == uri:
                    return b
            raise KeyError(uri)
        return self.benchmark_values[0]

    def __repr__(self):
        return str(self.name)


class MockBenchmark:
    """A mock Benchmark class."""

    def __init__(self, uri):
        self.uri = uri

    def __repr__(self):
        return str(self.name)


def test_seed_datasets_value():
    """Test that random seed is propagated to datasets."""
    da = MockDataset("a")
    db = MockDataset("b")
    datasets = Datasets((da, db))

    datasets.seed(123)

    for dataset in datasets:
        assert dataset.seed_value == 123

    assert da.seed_value == 123
    assert db.seed_value == 123


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


def test_enumerate_hidden_datasets():
    da = MockDataset("benchmark://a")
    db = MockDataset("benchmark://b")
    datasets = Datasets((da, db))

    db.hidden = True
    assert list(datasets) == [da]
    assert list(datasets.datasets(hidden=True)) == [da, db]


def test_enumerate_datasets_hidden_at_construction_time():
    da = MockDataset("benchmark://a")
    db = MockDataset("benchmark://b")
    db.hidden = True
    datasets = Datasets((da, db))

    assert list(datasets) == [da]
    assert list(datasets.datasets(hidden=True)) == [da, db]


def test_datasets_add_dataset():
    datasets = Datasets([])

    da = MockDataset("benchmark://foo-v0")
    datasets["benchmark://foo-v0"] = da

    assert list(datasets) == [da]


def test_datasets_add_hidden_dataset():
    datasets = Datasets([])

    da = MockDataset("benchmark://a")
    da.hidden = True
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


def test_datasets_get_item_default_protocol():
    da = MockDataset("benchmark://foo-v0")
    datasets = Datasets([da])

    assert datasets.dataset("foo-v0") == da
    assert datasets["foo-v0"] == da


def test_datasets_get_item_lookup_miss():
    da = MockDataset("benchmark://foo-v0")
    datasets = Datasets([da])

    with pytest.raises(LookupError) as e_ctx:
        datasets.dataset("benchmark://bar-v0")
    assert str(e_ctx.value) == "Dataset not found: benchmark://bar-v0"

    with pytest.raises(LookupError) as e_ctx:
        _ = datasets["benchmark://bar-v0"]
    assert str(e_ctx.value) == "Dataset not found: benchmark://bar-v0"


def test_dataset_empty():
    datasets = Datasets([])

    with pytest.raises(ValueError) as e_ctx:
        datasets.dataset()

    assert str(e_ctx.value) == "No datasets"


def test_benchmark_empty():
    datasets = Datasets([])

    with pytest.raises(ValueError) as e_ctx:
        datasets.benchmark()

    assert str(e_ctx.value) == "No datasets"


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


def test_benchmark_select_randomly():
    da = MockDataset("benchmark://foo-v0")
    db = MockDataset("benchmark://bar-v0")
    ba = MockBenchmark(uri="benchmark://foo-v0/abc")
    bb = MockBenchmark(uri="benchmark://bar-v0/abc")
    da.benchmark_values.append(ba)
    db.benchmark_values.append(bb)
    datasets = Datasets([da, db])

    # Create three lists of randomly selected benchmarks. Two using the same
    # seed, the third using a different seed. It is unlikely that the two
    # different seeds will produce the same lists.
    datasets.seed(1)
    benchmarks_a = [datasets.benchmark() for i in range(50)]
    datasets.seed(1)
    benchmarks_b = [datasets.benchmark() for i in range(50)]
    datasets.seed(2)
    benchmarks_c = [datasets.benchmark() for i in range(50)]

    assert benchmarks_a == benchmarks_b
    assert benchmarks_a != benchmarks_c
    assert len(set(benchmarks_a)) == 2


def test_dataset_select_randomly():
    da = MockDataset("benchmark://foo-v0")
    db = MockDataset("benchmark://bar-v0")
    datasets = Datasets([da, db])

    # Create three lists of randomly selected datasets. Two using the same seed,
    # the third using a different seed. It is unlikely that the two different
    # seeds will produce the same lists.
    datasets.seed(1)
    datasets_a = [datasets.dataset() for i in range(50)]
    datasets.seed(1)
    datasets_b = [datasets.dataset() for i in range(50)]
    datasets.seed(2)
    datasets_c = [datasets.dataset() for i in range(50)]

    assert datasets_a == datasets_b
    assert datasets_a != datasets_c
    assert len(set(datasets_a)) == 2


if __name__ == "__main__":
    main()
