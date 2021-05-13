# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/service/runtime:benchmark_cache."""

import pytest

from compiler_gym.service.proto import Benchmark, File
from compiler_gym.service.runtime.benchmark_cache import BenchmarkCache
from tests.test_main import main


def make_benchmark_of_size(size_in_bytes: int, target: int = 0) -> Benchmark:
    """Test helper. Generate a benchmark of the given size in bytes."""
    target = target or size_in_bytes
    bm = Benchmark(program=File(contents=("." * target).encode("utf-8")))
    size_offset = bm.ByteSize() - size_in_bytes
    if size_offset:
        return make_benchmark_of_size(size_in_bytes, size_in_bytes - size_offset)
    return bm


@pytest.mark.parametrize("size", [5, 10, 100, 1024])
def test_make_benchmark_of_size(size: int):
    """Sanity check for test helper function."""
    assert make_benchmark_of_size(size).ByteSize() == size


def test_oversized_benchmark_triggers_evict_to_capacity(mocker):
    cache = BenchmarkCache(max_size_in_bytes=10)

    mocker.spy(cache, "evict_to_capacity")

    cache["test"] = make_benchmark_of_size(50)

    assert cache.size == 1
    assert cache.size_in_bytes == 50

    cache.evict_to_capacity.assert_called_once()


def test_replace_existing_item():
    cache = BenchmarkCache()

    cache["a"] = make_benchmark_of_size(30)
    assert cache.size == 1
    assert cache.size_in_bytes == 30

    cache["a"] = make_benchmark_of_size(50)
    assert cache.size == 1
    assert cache.size_in_bytes == 50


def test_evict_to_capacity_on_max_size_reached(mocker):
    """Test that cache is evict_to_capacityd when the maximum size is exceeded."""
    cache = BenchmarkCache(max_size_in_bytes=100)

    mocker.spy(cache, "evict_to_capacity")
    mocker.spy(cache.logger, "info")

    cache["a"] = make_benchmark_of_size(30)
    cache["b"] = make_benchmark_of_size(30)
    cache["c"] = make_benchmark_of_size(30)
    assert cache.evict_to_capacity.call_count == 0

    cache["d"] = make_benchmark_of_size(30)
    assert cache.evict_to_capacity.call_count == 1

    assert cache.size == 2
    assert cache.size_in_bytes == 60

    cache.logger.info.assert_called_once_with(
        "Evicted %d benchmarks from cache. Benchmark cache size now %d bytes, "
        "%d items",
        2,
        30,
        1,
    )


def test_oversized_benchmark_emits_warning(mocker):
    """Test that a warning is emitted when a single item is larger than the
    entire target cache size.
    """
    cache = BenchmarkCache(max_size_in_bytes=10)

    mocker.spy(cache.logger, "warning")

    cache["test"] = make_benchmark_of_size(50)

    cache.logger.warning.assert_called_once_with(
        "Adding new benchmark with size %d bytes exceeds total target cache "
        "size of %d bytes",
        50,
        10,
    )


def test_contains():
    cache = BenchmarkCache(max_size_in_bytes=100)

    cache["a"] = make_benchmark_of_size(30)

    assert "a" in cache
    assert "b" not in cache


def test_getter():
    cache = BenchmarkCache(max_size_in_bytes=100)

    a = make_benchmark_of_size(30)
    b = make_benchmark_of_size(40)

    cache["a"] = a
    cache["b"] = b

    assert cache["a"] == a
    assert cache["a"] != b
    assert cache["b"] == b

    with pytest.raises(KeyError, match="c"):
        cache["c"]


def test_evict_to_capacity_on_maximum_size_update(mocker):
    """Test that cache is evict_to_capacityd when the maximum size is exceeded."""
    cache = BenchmarkCache(max_size_in_bytes=100)

    mocker.spy(cache, "evict_to_capacity")
    mocker.spy(cache.logger, "info")

    cache["a"] = make_benchmark_of_size(30)
    cache["b"] = make_benchmark_of_size(30)
    cache["c"] = make_benchmark_of_size(30)
    assert cache.evict_to_capacity.call_count == 0

    cache.max_size_in_bytes = 50
    assert cache.evict_to_capacity.call_count == 1
    assert cache.size_in_bytes == 30


if __name__ == "__main__":
    main()
