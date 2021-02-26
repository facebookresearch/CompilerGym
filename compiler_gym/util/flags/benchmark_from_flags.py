# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""A consistent way to interpret a user-specified benchmark from commandline flags."""
from typing import Optional, Union

from absl import flags

from compiler_gym.service.proto import Benchmark, File

flags.DEFINE_string(
    "benchmark",
    None,
    "The URI of the benchmark to use. Use the benchmark:// protocol to "
    "reference named benchmarks, or the file:/// protocol to reference paths "
    "to program data. If no protocol is specified, benchmark:// is implied.",
)

FLAGS = flags.FLAGS


def benchmark_from_flags() -> Optional[Union[Benchmark, str]]:
    """Returns either the name of the benchmark, or a Benchmark message."""
    if FLAGS.benchmark:
        if FLAGS.benchmark.startswith("file:///"):
            return Benchmark(uri=FLAGS.benchmark, program=File(uri=FLAGS.benchmark))
        else:
            return FLAGS.benchmark
    else:
        # No benchmark was specified.
        return None
