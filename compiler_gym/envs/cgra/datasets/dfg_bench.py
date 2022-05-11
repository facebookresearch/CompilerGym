# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import enum
import io
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import numpy as np
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional

import fasteners
from compiler_gym.datasets.dataset import Dataset

from compiler_gym.datasets import Benchmark, TarDatasetWithManifest
from compiler_gym.datasets.benchmark import ValidationCallback
from compiler_gym.datasets.uri import BenchmarkUri
from compiler_gym.envs.llvm import llvm_benchmark
from compiler_gym.errors import ValidationError
from compiler_gym.service.proto import BenchmarkDynamicConfig, Command
from compiler_gym.third_party import llvm
from compiler_gym.util.commands import Popen
from compiler_gym.util.download import download
from compiler_gym.util.runfiles_path import cache_path, site_data_path
from compiler_gym.util.timer import Timer

from compiler_gym.envs.cgra.Operations import Operations
from compiler_gym.envs.cgra.DFG import generate_DFG
import pickle

class GeneratedDFGs(Dataset):
    def __init__(self, size: int, site_data_base=None):
        super().__init__(
            "benchmark://dfg_" + str(size),
            "A dataset of automatically generated DFGs of a particular size.",
            "None",
            site_data_base=site_data_base
        )

        self.dfg_size = size

    def benchmark_uris_without_index(self):
        return "benchmark://dfg_" + str(self.dfg_size) + "/"

    def benchmark_uris(self) -> Iterable[str]:
        ind = 0
        while True:
            yield (self.benchmark_uris_without_index() + str(ind))
            ind += 1

    def benchmark_from_index(self, dfg_index, uri):
        dfg = generate_DFG(Operations, self.dfg_size, seed=dfg_index)

        return Benchmark.from_file_contents(uri=uri, data=pickle.dumps(dfg))

    def benchmark_from_parsed_uri(self, uri: BenchmarkUri) -> Benchmark:
        dfg_index = int(uri.path[1:])
        return self.benchmark_from_index(dfg_index, uri)

    def _random_benchmark(self, random_state: np.random.Generator) -> Benchmark:
        index = random_state.randomint(10000000000)
        return self.benchmark_from_index(index, self.benchmark_uris_without_index() + str(index))

class GeneratedDFGs5(GeneratedDFGs):
    def __init__(self, site_data_base=None):
        super().__init__(5, site_data_base)
class GeneratedDFGs10(GeneratedDFGs):
    def __init__(self, site_data_base=None):
        super().__init__(10, site_data_base)
class GeneratedDFGs15(GeneratedDFGs):
    def __init__(self, site_data_base=None):
        super().__init__(15, site_data_base)
class GeneratedDFGs20(GeneratedDFGs):
    def __init__(self, site_data_base=None):
        super().__init__(20, site_data_base)