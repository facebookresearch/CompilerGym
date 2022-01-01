// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "header.h"

#ifndef N
#define N 1000000
#endif

int A[N];
int B[N];

__attribute__((noinline)) void add(int* ret) {
  for (int i = 0; i < N; i++) A[i] = A[i] + B[i];

  *ret = A[N - 1];
}

__attribute__((optnone)) int main(int argc, char* argv[]) {
  int dummy = 0;
  // TODO: initialize tensors
  BENCH("add", add(&dummy), 100, dummy);

  return 0;
}
