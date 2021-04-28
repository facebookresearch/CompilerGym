# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/datasets:benchmark."""
import re
from pathlib import Path

import pytest

from compiler_gym.datasets import Benchmark, BenchmarkSource
from compiler_gym.datasets.uri import BENCHMARK_URI_RE, DATASET_NAME_RE
from compiler_gym.service.proto import Benchmark as BenchmarkProto
from compiler_gym.validation_error import ValidationError
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.common"]


def _rgx_match(regex, groupname, string) -> str:
    """Match the regex and return a named group."""
    match = re.match(regex, string)
    assert match, f"Failed to match regex '{regex}' using string '{groupname}'"
    return match.group(groupname)


def test_benchmark_uri_protocol():
    assert (
        _rgx_match(DATASET_NAME_RE, "dataset_protocol", "benchmark://cbench-v1/")
        == "benchmark"
    )
    assert (
        _rgx_match(DATASET_NAME_RE, "dataset_protocol", "Generator13://gen-v11/")
        == "Generator13"
    )


def test_invalid_benchmark_uris():
    # Invalid protocol
    assert not DATASET_NAME_RE.match("B?://cbench-v1/")  # Invalid characters
    assert not DATASET_NAME_RE.match("cbench-v1/")  # Missing protocol

    # Invalid dataset name
    assert not BENCHMARK_URI_RE.match("benchmark://cbench?v0/foo")  # Invalid character
    assert not BENCHMARK_URI_RE.match(
        "benchmark://cbench/foo"
    )  # Missing version suffix
    assert not BENCHMARK_URI_RE.match("benchmark://cbench-v0")  # Missing benchmark ID
    assert not BENCHMARK_URI_RE.match("benchmark://cbench-v0/")  # Missing benchmark ID

    # Invalid benchmark ID
    assert not BENCHMARK_URI_RE.match("benchmark://cbench-v1/ whitespace")  # Whitespace
    assert not BENCHMARK_URI_RE.match("benchmark://cbench-v1/\t")  # Whitespace


def test_benchmark_uri_dataset():
    assert (
        _rgx_match(BENCHMARK_URI_RE, "dataset_name", "benchmark://cbench-v1/foo")
        == "cbench-v1"
    )
    assert (
        _rgx_match(BENCHMARK_URI_RE, "dataset_name", "Generator13://gen-v11/foo")
        == "gen-v11"
    )


def test_benchmark_dataset_name():
    assert (
        _rgx_match(BENCHMARK_URI_RE, "dataset", "benchmark://cbench-v1/foo")
        == "benchmark://cbench-v1"
    )
    assert (
        _rgx_match(BENCHMARK_URI_RE, "dataset", "Generator13://gen-v11/foo")
        == "Generator13://gen-v11"
    )


def test_benchmark_uri_id():
    assert (
        _rgx_match(BENCHMARK_URI_RE, "benchmark_name", "benchmark://cbench-v1/foo")
        == "foo"
    )
    assert (
        _rgx_match(BENCHMARK_URI_RE, "benchmark_name", "benchmark://cbench-v1/foo/123")
        == "foo/123"
    )
    assert (
        _rgx_match(
            BENCHMARK_URI_RE, "benchmark_name", "benchmark://cbench-v1/foo/123.txt"
        )
        == "foo/123.txt"
    )
    assert (
        _rgx_match(
            BENCHMARK_URI_RE,
            "benchmark_name",
            "benchmark://cbench-v1/foo/123?param=true&false",
        )
        == "foo/123?param=true&false"
    )


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
    path.touch()
    benchmark = Benchmark.from_file("benchmark://example-v0/foo", path)
    # Use startswith() and endswith() because macOS can add a /private prefix to
    # paths.
    assert benchmark.proto.program.uri.startswith("file:///")
    assert benchmark.proto.program.uri.endswith(str(path))


def test_benchmark_from_file_not_found(tmpwd: Path):
    path = tmpwd / "foo.txt"
    with pytest.raises(FileNotFoundError) as e_ctx:
        Benchmark.from_file("benchmark://example-v0/foo", path)

    # Use  endswith() because macOS can add a /private prefix to paths.
    assert str(e_ctx.value).endswith(str(path))


if __name__ == "__main__":
    main()
