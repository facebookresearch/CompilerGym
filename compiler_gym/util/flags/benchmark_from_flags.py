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
    "The URI of the benchmark to use.",
)
flags.DEFINE_string(
    "program_data",
    None,
    "The URI of a file containing the program data to use. Paths to local "
    "files are representing in the format file:///absolute/path/to/file",
)

FLAGS = flags.FLAGS


def benchmark_from_flags() -> Optional[Union[Benchmark, str]]:
    """Returns either the name of the benchmark, or a Benchmark message."""
    if FLAGS.benchmark:
        return FLAGS.benchmark
    elif FLAGS.program_data:
        return Benchmark(uri=FLAGS.program_data, program=File(uri=FLAGS.program_data))
    else:
        # No benchmark was specified.
        return None
