# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for compiler_gym.datasets.uri."""
from compiler_gym.datasets import BenchmarkUri
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.common"]


def test_from_string_1():
    uri = BenchmarkUri.from_string("benchmark://test-v0")
    assert uri.scheme == "benchmark"
    assert uri.dataset == "test-v0"
    assert uri.path == ""
    assert uri.params == {}
    assert uri.fragment == ""
    assert str(uri) == "benchmark://test-v0"


def test_from_string_2():
    uri = BenchmarkUri.from_string("test-v0")
    assert uri.scheme == "benchmark"
    assert uri.dataset == "test-v0"
    assert uri.path == ""
    assert uri.params == {}
    assert uri.fragment == ""
    assert str(uri) == "benchmark://test-v0"


def test_from_string_3():
    uri = BenchmarkUri.from_string("benchmark://test-v0")
    assert uri.scheme == "benchmark"
    assert uri.dataset == "test-v0"
    assert uri.path == ""
    assert uri.params == {}
    assert uri.fragment == ""
    assert str(uri) == "benchmark://test-v0"


def test_from_string_4():
    uri = BenchmarkUri.from_string(
        "generator://csmith-v0/this path has whitespace/in/it"
    )
    assert uri.scheme == "generator"
    assert uri.dataset == "csmith-v0"
    assert uri.path == "/this path has whitespace/in/it"
    assert uri.params == {}
    assert uri.fragment == ""
    assert str(uri) == "generator://csmith-v0/this path has whitespace/in/it"


def test_from_string_5():
    uri = BenchmarkUri.from_string("generator://csmith-v0/0")
    assert uri.scheme == "generator"
    assert uri.dataset == "csmith-v0"
    assert uri.path == "/0"
    assert uri.params == {}
    assert uri.fragment == ""
    assert str(uri) == "generator://csmith-v0/0"


def test_from_string_6():
    uri = BenchmarkUri.from_string("generator://csmith-v0?a=b&c=d#foo")
    assert uri.scheme == "generator"
    assert uri.dataset == "csmith-v0"
    assert uri.path == ""
    assert uri.params == {"a": ["b"], "c": ["d"]}
    assert uri.fragment == "foo"
    assert str(uri) == "generator://csmith-v0?a=b&c=d#foo"


def test_from_string_7():
    uri = BenchmarkUri.from_string("")
    assert uri.scheme == "benchmark"
    assert uri.dataset == ""
    assert uri.path == ""
    assert uri.params == {}
    assert uri.fragment == ""
    assert str(uri) == "benchmark:"


def test_from_string_8():
    uri = BenchmarkUri.from_string("generator:")
    assert uri.scheme == "generator"
    assert uri.dataset == ""
    assert uri.path == ""
    assert uri.params == {}
    assert uri.fragment == ""
    assert str(uri) == "generator:"


def test_canonicalize_1():
    assert BenchmarkUri.canonicalize("test-v0") == "benchmark://test-v0"


if __name__ == "__main__":
    main()
