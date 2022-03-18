# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from pathlib import Path
from typing import Iterable, Optional

from compiler_gym.datasets import Benchmark, BenchmarkSource, Dataset
from compiler_gym.datasets.benchmark import BenchmarkWithSource

# from compiler_gym.envs.mlir.mlir_benchmark import ClangInvocation
from compiler_gym.service.proto import BenchmarkDynamicConfig, Command
from compiler_gym.util.decorators import memoized_property
from compiler_gym.util.runfiles_path import runfiles_path, site_data_path
from compiler_gym.util.shell_format import plural

logger = logging.getLogger(__name__)

# The maximum value for the --seed argument to matmul.
UINT_MAX = (2 ** 32) - 1

_matmul_BIN = runfiles_path("compiler_gym/third_party/matmul/matmul/bin/matmul")
_matmul_INCLUDES = runfiles_path(
    "compiler_gym/third_party/matmul/matmul/include/matmul-2.3.0"
)

matmul_sizes = [
    (64, 64, 64)
    # (128, 128, 128),
    # (1024, 1024, 1024)
]


class MatmulBenchmark(BenchmarkWithSource):
    """A matmul benchmark."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._src = None
        self.proto.dynamic_config.MergeFrom(
            BenchmarkDynamicConfig(
                build_cmd=Command(
                    argument=["$CC", "$IN"],
                    outfile=["benchmark_main"],
                    timeout_seconds=60,
                ),
                run_cmd=Command(
                    argument=["./benchmark_main", "--benchmark_format=json"],
                    timeout_seconds=300,
                ),
            )
        )

    @classmethod
    def create(cls, uri: str, bitcode: bytes, src: bytes) -> Benchmark:
        """Create a benchmark from paths."""
        benchmark = cls.from_file_contents(uri, bitcode)
        benchmark._src = src  # pylint: disable=protected-access
        return benchmark

    @memoized_property
    def sources(self) -> Iterable[BenchmarkSource]:
        return [
            BenchmarkSource(filename="source.mlir", contents=self._src),
        ]

    @property
    def source(self) -> str:
        """Return the single source file contents as a string."""
        return self._src.decode("utf-8")


class MatmulDataset(Dataset):
    """A dataset which generates matmul programs."""

    def __init__(
        self,
        site_data_base: Path,
        sort_order: int = 0,
        matmul_bin: Optional[Path] = None,
        matmul_includes: Optional[Path] = None,
    ):
        """Constructor.
        :param site_data_base: The base path of a directory that will be used to
            store installed files.
        :param sort_order: An optional numeric value that should be used to
            order this dataset relative to others. Lowest value sorts first.
        :param matmul_bin: The path of the matmul binary to use. If not
            provided, the version of matmul shipped with CompilerGym is used.
        :param matmul_includes: The path of the matmul includes directory. If
            not provided, the includes of the matmul shipped with CompilerGym is
            used.
        """
        super().__init__(
            name="generator://matmul-v0",
            description="Targeted size matmul programs",
            references={},
            license="donotsubmit",
            site_data_base=site_data_base,
            sort_order=sort_order,
            benchmark_class=MatmulBenchmark,
        )
        self.matmul_bin_path = matmul_bin or _matmul_BIN
        self.matmul_includes_path = matmul_includes or _matmul_INCLUDES
        """# The command that is used to compile an mlir-IR bitcode file from a
        # matmul input. Reads from stdin, writes to stdout.
        self.clang_compile_command: List[str] = ClangInvocation.from_c_file(
            "-",  # Read from stdin.
            copt=[
                "-xc",  # The C programming language.
                "-ferror-limit=1",  # Stop on first error.
                "-w",  # No warnings.
                f"-I{self.matmul_includes_path}",  # Include the matmul headers.
            ],
        ).command(
            outpath="-"  # Write to stdout.
        )"""

    @property
    def size(self) -> int:
        # Actually 2^32 - 1, but practically infinite for all intents and
        # purposes.
        return len(matmul_sizes)

    def name_from_size(self, mnk):
        return f"{self.name}/{mnk[0]}_{mnk[1]}_{mnk[2]}"

    # TODO(kyleherndon): Benchmarks are actually dynamically generated for any
    # provided parameters, figure out a better way to represent this in the list of
    # available benchmarks
    def benchmark_uris(self) -> Iterable[str]:
        return (self.name_from_size(mnk) for mnk in matmul_sizes)

    def benchmark(self, uri: str) -> MatmulBenchmark:
        sizestr = uri.split("/")[-1]
        sizetuple = (int(i) for i in sizestr.split("_"))
        return self.benchmark_from_size(sizetuple)

    def benchmark_from_size(
        self, mnk, max_retries: int = 3, retry_count: int = 0
    ) -> MatmulBenchmark:
        """Get a benchmark from a uint32 seed.
        :param mnk: 3-tuple containing m, n, k sizes of the matmul
        :return: A benchmark instance.
        :raises OSError: If matmul fails.
        :raises BenchmarkInitError: If the C program generated by matmul cannot
            be lowered to mlir-IR.
        """
        if retry_count >= max_retries:
            raise OSError(
                f"matmul failed after {retry_count} {plural(retry_count, 'attempt', 'attempts')} "
                f"with size {mnk}"
            )

        self.install()
        mnk = list(mnk)
        # Run matmul with the given size and regex to produce the correct mlir
        logger.debug("Exec matmul --mnk %d", mnk)
        """matmul = subprocess.Popen(
            [str(self.matmul_bin_path), "--mnk", str(mnk)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )"""

        # TODO(kyleherndon): refactor these to another location
        src_content = """
func @matmul(%a: tensor<${M}x${K}xf32> {linalg.buffer_layout = affine_map<(i, j)[s0, s1] -> (i, j)>},
             %b: tensor<${K}x${N}xf32> {linalg.buffer_layout = affine_map<(i, j)[s0, s1] -> (i, j)>},
             %c: tensor<${M}x${N}xf32> {linalg.buffer_layout = affine_map<(i, j)[s0, s1] -> (i, j)>}) -> tensor<${M}x${N}xf32>
attributes { passthrough = [["target-cpu", "haswell"], ["prefer-vector-width", "256"]]}
{
  %f0 = arith.constant 0.0 : f32
  %f1 = linalg.fill(%f0, %c) : f32, tensor<${M}x${N}xf32> -> tensor<${M}x${N}xf32>
  %d = linalg.matmul ins(%a, %b : tensor<${M}x${K}xf32>, tensor<${K}x${N}xf32>)
    outs(%f1: tensor<${M}x${N}xf32>) -> tensor<${M}x${N}xf32>
  return %d : tensor<${M}x${N}xf32>
}"""
        cc_src = """
#include <benchmark/benchmark.h>
#include <mlir/ExecutionEngine/RunnerUtils.h>

#include <cstdio>
#include <vector>

void naive_matmul(const float* a, const float* b, float* c, size_t m, size_t k, size_t n) {
  // correctness check
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
#ifdef COLUMN_MAJOR
      size_t ci = i + j * m;
#else
      size_t ci = i * n + j;
#endif
      c[ci] = 0.0f;
      for (size_t p = 0; p < k; p++) {
#ifdef COLUMN_MAJOR
        c[ci] += a[i + p * m] * b[p + j * k];
#else
        c[ci] += a[i * k + p] * b[p * n + j];
#endif
      }
    }
  }
}

void init_matrix(float* a, int nrows, int ncols) {
  for (int j = 0; j < ncols; j++) {
    for (int i = 0; i < nrows; i++) {
      a[i + j * nrows] = ((float)rand() / (float)RAND_MAX);
    }
  }
}

extern "C" {
void matmul(float* aligned_a, float* allocated_a, int64_t offset_a, int64_t size_a0,
            int64_t size_a1, int64_t strides_a0, int64_t strides_a1, float* aligned_b,
            float* allocated_b, int64_t offset_b, int64_t size_b0, int64_t size_b1,
            int64_t strides_b0, int64_t strides_b1, float* aligned_c, float* allocated_c,
            int64_t offset_c, int64_t size_c0, int64_t size_c1, int64_t strides_c0,
            int64_t strides_c1);
}

size_t g_errors = 0;
static void BenchmarkFunction(benchmark::State& state) {
  // TODO(boian): pass these as command line arguments
  int MDIM = ${M};
  int NDIM = ${N};
  int KDIM = ${K};
  std::vector<float> a(MDIM * KDIM);
  std::vector<float> b(KDIM * NDIM);
  std::vector<float> c(MDIM * NDIM);
  float *A = a.data(), *B = b.data(), *C = c.data();
  //  a[0] = 1; b[0] = 2;
  init_matrix(A, MDIM, KDIM);
  init_matrix(B, KDIM, NDIM);
  init_matrix(C, MDIM, NDIM);
  int LDA = KDIM;
  int LDB = NDIM;
  int LDC = NDIM;

  for (auto _ : state) {
    matmul(A, A, 0, MDIM, KDIM, LDA, 1, B, B, 0, KDIM, NDIM, LDB, 1, C, C, 0, MDIM, NDIM, LDC, 1);
  }

  std::vector<float> c2(MDIM * NDIM);
  float* C2 = c2.data();
  size_t errors = 0;
  naive_matmul(A, B, C2, MDIM, KDIM, NDIM);
  for (size_t i = 0; i < MDIM; i++) {
    for (size_t j = 0; j < NDIM; j++) {
      size_t ci = i + j * MDIM;
      if (std::abs(C[ci] - C2[ci]) > 0.01f) {
        if (errors == 0) {
          fprintf(stderr, "Incorrect result at index %ld,%ld: C=%0.2f C2=%0.2f\\n", i, j, C[ci],
                  C2[ci]);
        }
        errors++;
      }
    }
  }
  fprintf(stderr, "Detected %ld errors.\\n", errors);
  g_errors = errors;
}

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  benchmark::RegisterBenchmark("BM_Matmul", BenchmarkFunction)
      ->MeasureProcessCPUTime()
      ->UseRealTime();
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return g_errors != 0;
}
"""
        mlir_site_dir = site_data_path("mlir-v0")
        mlir_site_dir.mkdir(parents=True, exist_ok=True)
        mlir_file_path = site_data_path("mlir-v0") / "matmul.mlir.template"
        with open(mlir_file_path, "w+") as mlir_file:
            mlir_file.write(src_content)
            mlir_file.close()
        cc_file_path = site_data_path("mlir-v0") / "benchmark_main.cc.template"
        with open(cc_file_path, "w+") as cc_file:
            cc_file.write(cc_src)
            cc_file.close()
        new_content = src_content.replace("${M}", str(mnk[0]))
        new_content = new_content.replace("${N}", str(mnk[1]))
        content = new_content.replace("${K}", str(mnk[2]))

        return self.benchmark_class.create(
            self.name_from_size(mnk),
            bytes(content, "utf-8"),
            bytes(src_content, "utf-8"),
        )
