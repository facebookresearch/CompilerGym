# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Smoke test for examples/example_unrolling_service/example_without_bazel.py"""
from example_unrolling_service.example_without_bazel import main
from flaky import flaky


@flaky
def test_demo_without_bazel():
    main()
