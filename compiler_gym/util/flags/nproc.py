# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from multiprocessing import cpu_count

from absl import flags

flags.DEFINE_integer("nproc", cpu_count(), "The number of parallel processes to run.")
