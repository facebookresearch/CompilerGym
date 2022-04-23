# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class BenchmarkInitError(OSError):
    """Base class for errors raised if a benchmark fails to initialize."""


class DatasetInitError(OSError):
    """Base class for errors raised if a dataset fails to initialize."""
