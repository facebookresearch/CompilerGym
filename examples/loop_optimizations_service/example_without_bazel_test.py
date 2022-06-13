# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Smoke test for examples/loop_optimizations_service/example_without_bazel.py"""
from flaky import flaky
from loop_optimizations_service.example_without_bazel import main


@flaky
def test_example_without_bazel():
    main()
