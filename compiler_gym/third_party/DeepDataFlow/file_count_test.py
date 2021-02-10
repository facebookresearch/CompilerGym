# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Test that the DeepDataFlow dataset contains the expected numbers of files."""
import pytest

from compiler_gym.util.runfiles_path import runfiles_path
from tests.test_main import main

# The number of bitcode files in the DeepDataFlow dataset, grouped by source.
EXPECTED_NUMBER_OF_BITCODE_FILES = {
    "blas": 300,
    "linux": 13920,
    "github": 50708,
    "npb": 122,
    "poj104": 49628,
    "tensorflow": 1985,
}


@pytest.fixture(scope="session", params=list(EXPECTED_NUMBER_OF_BITCODE_FILES.keys()))
def subset(request):
    return request.param


def test_deep_dataflow_file_count(subset: str):
    bitcode_dir = runfiles_path("compiler_gym/third_party/DeepDataFlow") / subset
    num_files = len([f for f in bitcode_dir.iterdir() if f.name.endswith(".bc")])
    assert num_files == EXPECTED_NUMBER_OF_BITCODE_FILES[subset]


if __name__ == "__main__":
    main()
