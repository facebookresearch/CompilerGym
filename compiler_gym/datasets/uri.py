# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module contains utility code for working with URIs."""
import re

# Regular expression that matches the full two-part URI prefix of a dataset:
#     {{protocol}}://{{dataset}}
#
# An optional trailing slash is permitted.
#
# Example matches: "benchmark://foo-v0", "generator://bar-v0/".
DATASET_NAME_PATTERN = r"(?P<dataset>(?P<dataset_protocol>[a-zA-z0-9-_]+)://(?P<dataset_name>[a-zA-z0-9-_]+-v(?P<dataset_version>[0-9]+)))/?"
DATASET_NAME_RE = re.compile(DATASET_NAME_PATTERN)

# Regular expression that matches the full three-part format of a benchmark URI:
#     {{protocol}}://{{dataset}}/{{id}}
#
# Example matches: "benchmark://foo-v0/foo" or "generator://bar-v1/foo/bar.txt".
BENCHMARK_URI_PATTERN = r"(?P<dataset>(?P<dataset_protocol>[a-zA-z0-9-_]+)://(?P<dataset_name>[a-zA-z0-9-_]+-v(?P<dataset_version>[0-9]+)))/(?P<benchmark_name>[^\s]+)$"
BENCHMARK_URI_RE = re.compile(BENCHMARK_URI_PATTERN)


def resolve_uri_protocol(uri: str) -> str:
    """Require that the URI has a protocol by applying a default "benchmark"
    protocol if none is set."""
    if "://" not in uri:
        return f"benchmark://{uri}"
    return uri
