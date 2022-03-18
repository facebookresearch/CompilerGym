#include <cstdlib>
#include <fstream>
#include <iostream>

#include "mlir/ExecutionEngine/RunnerUtils.h"
void init_matrix(float* a, int nrows, int ncols) {
  for (int j = 0; j < ncols; j++) {
    for (int i = 0; i < nrows; i++) {
      a[i + j * nrows] = ((float)rand() / (float)RAND_MAX);
    }
  }
}
\n extern "C" {
  void matmul(float* aligned_a, float* allocated_a, int64_t offset_a, int64_t size_a0,
              int64_t size_a1, int64_t strides_a0, int64_t strides_a1, float* aligned_b,
              float* allocated_b, int64_t offset_b, int64_t size_b0, int64_t size_b1,
              int64_t strides_b0, int64_t strides_b1, float* aligned_c, float* allocated_c,
              int64_t offset_c, int64_t size_c0, int64_t size_c1, int64_t strides_c0,
              int64_t strides_c1);
}
int main(int argc, char** argv) {
  int MDIM = 64;
  int NDIM = 64;
  int KDIM = 64;
  float *A, *B, *C;
  std::ofstream out;
  A = (float*)malloc(MDIM * KDIM * sizeof(float));
  B = (float*)malloc(KDIM * NDIM * sizeof(float));
  C = (float*)malloc(MDIM * NDIM * sizeof(float));
  init_matrix(A, MDIM, KDIM);
  init_matrix(B, KDIM, NDIM);
  init_matrix(C, MDIM, NDIM);
  int LDA = KDIM;
  int LDB = NDIM;
  int LDC = NDIM;
  matmul(A, A, 0, MDIM, KDIM, 1, LDA, B, B, 0, KDIM, NDIM, 1, LDB, C, C, 0, MDIM, NDIM, 1, LDC);
  out.close();
  return 0;
}
