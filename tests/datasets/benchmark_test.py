# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/datasets:benchmark."""
from pathlib import Path

import pytest

from compiler_gym.datasets import Benchmark, BenchmarkSource
from compiler_gym.service.proto import Benchmark as BenchmarkProto
from compiler_gym.validation_error import ValidationError
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.common"]


def test_benchmark_attribute_outside_init():
    """Test that new attributes cannot be added to Benchmark."""
    benchmark = Benchmark(None)
    with pytest.raises(AttributeError):
        # pylint: disable=assigning-non-slot
        benchmark.foobar = 123  # noqa


def test_benchmark_subclass_attribute_outside_init():
    """Test that new attributes can be added to Benchmark subclass."""

    class TestBenchmark(Benchmark):
        pass

    benchmark = TestBenchmark(None)
    benchmark.foobar = 123  # pylint: disable=attribute-defined-outside-init
    assert benchmark.foobar == 123


def test_benchmark_properties():
    """Test benchmark properties."""
    benchmark = Benchmark(BenchmarkProto(uri="benchmark://example-v0/foobar"))
    assert benchmark.uri == "benchmark://example-v0/foobar"
    assert benchmark.proto == BenchmarkProto(uri="benchmark://example-v0/foobar")


def test_benchmark_immutable():
    """Test that benchmark properties are immutable."""
    benchmark = Benchmark(BenchmarkProto(uri="benchmark://example-v0/foobar"))
    with pytest.raises(AttributeError):
        benchmark.uri = 123
    with pytest.raises(AttributeError):
        benchmark.proto = 123


def test_add_validation_callbacks_values():
    """Test methods for adding and checking custom validation callbacks."""

    def a(env):
        pass

    benchmark = Benchmark(BenchmarkProto(uri="benchmark://example-v0/foobar"))
    assert benchmark.validation_callbacks() == []
    assert not benchmark.is_validatable()

    benchmark.add_validation_callback(a)
    assert benchmark.validation_callbacks() == [a]
    assert benchmark.is_validatable()

    benchmark.add_validation_callback(a)
    assert benchmark.validation_callbacks() == [a, a]


def test_add_validation_callbacks_call_count():
    """Test that custom validation callbacks are called on validate()."""
    a_call_count = 0
    b_call_count = 0

    def a(env):
        nonlocal a_call_count
        a_call_count += 1

    def b(env):
        nonlocal b_call_count
        b_call_count += 1

    benchmark = Benchmark(BenchmarkProto(uri="benchmark://example-v0/foobar"))
    benchmark.add_validation_callback(a)

    errors = benchmark.validate(env=None)
    assert errors == []
    assert a_call_count == 1
    assert b_call_count == 0

    benchmark.add_validation_callback(b)
    errors = benchmark.validate(env=None)
    assert errors == []
    assert a_call_count == 2
    assert b_call_count == 1


def test_validation_callback_error():
    """Test error propagation from custom validation callback."""

    def a(env):
        yield ValidationError(type="Compilation Error")
        yield ValidationError(type="Runtime Error")

    benchmark = Benchmark(BenchmarkProto(uri="benchmark://example-v0/foobar"))
    benchmark.add_validation_callback(a)

    errors = benchmark.validate(env=None)
    assert errors == [
        ValidationError(type="Compilation Error"),
        ValidationError(type="Runtime Error"),
    ]


def test_validation_callback_error_iter():
    """Test error propagation from custom validation callback using iterable."""

    def a(env):
        yield ValidationError(type="Compilation Error")
        yield ValidationError(type="Runtime Error")

    benchmark = Benchmark(BenchmarkProto(uri="benchmark://example-v0/foobar"))
    benchmark.add_validation_callback(a)

    errors = benchmark.ivalidate(env=None)
    next(errors) == ValidationError(type="Compilation Error")
    next(errors) == ValidationError(type="Runtime Error")


def test_validation_callback_flaky():
    """Test error propagation on callback which *may* fail."""
    flaky = False

    def a(env):
        nonlocal flaky
        del env
        if flaky:
            yield ValidationError(type="Runtime Error")

    benchmark = Benchmark(BenchmarkProto(uri="benchmark://example-v0/foobar"))
    benchmark.add_validation_callback(a)

    errors = benchmark.validate(env=None)
    assert errors == []

    flaky = True
    errors = benchmark.validate(env=None)
    assert errors == [
        ValidationError(type="Runtime Error"),
    ]


def test_eq_benchmarks():
    a = Benchmark(BenchmarkProto(uri="benchmark://example-v0/foo"))
    b = Benchmark(BenchmarkProto(uri="benchmark://example-v0/foo"))

    assert a == b


def test_eq_strings():
    a = Benchmark(BenchmarkProto(uri="benchmark://example-v0/foo"))
    b = "benchmark://example-v0/foo"

    assert a == b


def test_ne_benchmarks():
    a = Benchmark(BenchmarkProto(uri="benchmark://example-v0/foo"))
    b = Benchmark(BenchmarkProto(uri="benchmark://example-v0/bar"))

    assert a != b


def test_ne_strings():
    a = Benchmark(BenchmarkProto(uri="benchmark://example-v0/foo"))
    b = "benchmark://example-v0/bar"

    assert a != b


def test_benchmark_sources(tmpwd: Path):
    a = Benchmark(
        BenchmarkProto(uri="benchmark://example-v0/foo"),
        sources=[("example.py", "Hello, world!".encode("utf-8"))],
    )
    a.add_source(BenchmarkSource(filename="foo.py", contents="Hi".encode("utf-8")))

    assert list(a.sources) == [
        BenchmarkSource("example.py", "Hello, world!".encode("utf-8")),
        BenchmarkSource(filename="foo.py", contents="Hi".encode("utf-8")),
    ]

    a.write_sources_to_directory("benchmark_sources")

    with open(tmpwd / "benchmark_sources" / "example.py") as f:
        assert f.read() == "Hello, world!"
    with open(tmpwd / "benchmark_sources" / "foo.py") as f:
        assert f.read() == "Hi"


def test_benchmark_from_file(tmpwd: Path):
    path = tmpwd / "foo.txt"
    with open(path, "w") as f:
        f.write("Hello, world!")
    benchmark = Benchmark.from_file("benchmark://example-v0/foo", path)
    assert benchmark.proto.program.contents.decode("utf-8") == "Hello, world!"


def test_benchmark_from_file_not_found(tmpwd: Path):
    path = tmpwd / "foo.txt"
    with pytest.raises(FileNotFoundError, match=str(path)):
        Benchmark.from_file("benchmark://example-v0/foo", path)


def test_dataset_equality_and_sorting():
    """Test comparison operators between datasets."""
    a = Benchmark(BenchmarkProto(uri="benchmark://example-v0/a"))
    a2 = Benchmark(BenchmarkProto(uri="benchmark://example-v0/a"))
    b = Benchmark(BenchmarkProto(uri="benchmark://example-v0/b"))

    assert a == a2
    assert a != b
    assert a < b
    assert a <= b
    assert b > a
    assert b >= a

    # String comparisons
    assert a == "benchmark://example-v0/a"
    assert a != "benchmark://example-v0/b"
    assert a < "benchmark://example-v0/b"

    # Sorting
    assert sorted([a2, b, a]) == [
        "benchmark://example-v0/a",
        "benchmark://example-v0/a",
        "benchmark://example-v0/b",
    ]


if __name__ == "__main__":
    main()
