# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module contains utility code for working with URIs."""
from typing import Dict, List, Union
from urllib.parse import ParseResult, parse_qs, urlencode, urlparse, urlunparse

from pydantic import BaseModel


class BenchmarkUri(BaseModel):
    """A URI used to identify a benchmark, and optionally a set of parameters
    for the benchmark.

    A URI has the following format:

    .. code-block::

        scheme://dataset/path?params#fragment

    where:

    * :code:`scheme` (optional, default :code:`benchmark`): An arbitrary string
      used to group datasets, for example :code:`generator` if the dataset is a
      benchmark generator.

    * :code:`dataset`: The name of a dataset, optionally with a version tag, for
      example :code:`linux-v0`.

    * :code:`path` (optional, default empty string): The path of a benchmark
      within a dataset.

    * :code:`params` (optional, default empty dictionary): A set of query
      parameters for the benchmark. This is parsed a dictionary of string keys
      to a list of string values. For example :code:`dataset=1&debug=true` which
      will be parsed as :code:`{"dataset": ["1"], "debug": ["true"]}`.

    * :code:`fragment` (optional, default empty string): An optional fragment
      within the benchmark.

    The :code:`scheme` and :code:`dataset` components are used to resolve a
    :class:`Dataset <compiler_gym.datasets.Dataset>` class that can serve the
    benchmark. The :meth:`Dataset.benchmark_from_parsed_uri()` method is then
    used to interpret the remainder of the URI components.

    A benchmark URI may resolve to zero or more benchmarks, for example:

    * :code:`benchmark://csmith-v0` resolves to any benchmark from the
      :code:`benchmark://csmith-v0` dataset.

    * :code:`cbench-v0/qsort` resolves to the path :code:`/qsort`
      within the dataset :code:`benchmark://cbench-v0` using the default scheme.

    * :code:`benchmark://cbench-v0/qsort?debug=true` also resolves to the path
      :code:`/qsort` within the dataset :code:`benchmark://cbench-v0`, but with
      an additional parameter :code:`debug=true`.
    """

    scheme: str
    """The benchmark scheme. Defaults to :code:`benchmark`."""

    dataset: str
    """The name of the dataset."""

    path: str
    """The path of the benchmark. Empty string if not set."""

    params: Dict[str, List[str]] = {}
    """A dictionary of query parameters. Empty dictionary if not set."""

    fragment: str = ""
    """The URL fragment. Empty string if not set."""

    @staticmethod
    def canonicalize(uri: str):
        return str(BenchmarkUri.from_string(uri))

    @classmethod
    def from_string(cls, uri: str) -> "BenchmarkUri":
        components = urlparse(uri)

        # Add the default "benchmark://" scheme if required.
        if not components.scheme and not components.netloc:
            components = urlparse(f"benchmark://{uri}")

        return cls(
            scheme=components.scheme,
            dataset=components.netloc,
            path=components.path,
            params=parse_qs(components.query),
            fragment=components.fragment,
        )

    def startswith(self, *args):
        return str(self).startswith(*args)

    def endswith(self, *args):
        return str(self).endswith(*args)

    def __repr__(self):
        return urlunparse(
            ParseResult(
                scheme=self.scheme,
                netloc=self.dataset,
                path=self.path,
                query=urlencode(self.params, doseq=True),
                fragment=self.fragment,
                params="",  # Field not used.
            )
        )

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: Union["BenchmarkUri", str]) -> bool:
        return str(self) == str(other)

    def __lt__(self, other: Union["BenchmarkUri", str]) -> bool:
        return str(self) < str(other)
