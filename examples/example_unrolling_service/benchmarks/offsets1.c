// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "header.h"

#ifndef N
#define N 1000000
#endif

#ifndef n
#define n 3
#endif

int A[N];

__attribute__((noinline)) void example1(int* ret) {
  //#pragma unroll(n)
  for (int i = 0; i < N - 3; i++) A[i] = A[i + 1] + A[i + 2] + A[i + 3];

  *ret = A[N - 1];
}

__attribute__((optnone)) int main(int argc, char* argv[]) {
  int dummy = 0;
  // TODO: initialize tensors
  BENCH("example1", example1(&dummy), 100, dummy);

  return 0;
}
