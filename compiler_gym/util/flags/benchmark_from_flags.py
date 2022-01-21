# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""A consistent way to interpret a user-specified benchmark from commandline flags."""
from pathlib import Path
from typing import Optional, Union

from absl import flags

from compiler_gym.datasets import Benchmark

flags.DEFINE_string(
    "benchmark",
    None,
    "The URI of the benchmark to use. Use the benchmark:// scheme to "
    "reference named benchmarks, or the file:/// scheme to reference paths "
    "to program data. If no scheme is specified, benchmark:// is implied.",
)

FLAGS = flags.FLAGS


def benchmark_from_flags() -> Optional[Union[Benchmark, str]]:
    """Returns either the name of the benchmark, or a Benchmark message."""
    if FLAGS.benchmark:
        if FLAGS.benchmark.startswith("file:///"):
            path = Path(FLAGS.benchmark[len("file:///") :])
            uri = f"benchmark://user-v0/{path}"
            return Benchmark.from_file(uri=uri, path=path)
        else:
            return FLAGS.benchmark
    else:
        # No benchmark was specified.
        return None
