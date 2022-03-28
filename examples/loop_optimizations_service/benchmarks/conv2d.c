// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "header.h"

// TODO: use templates instead of macros
#ifndef N
#define N 32
#endif

#ifndef Ih
#define Ih 3
#endif

#ifndef Iw
#define Iw 12
#endif

#ifndef Ic
#define Ic 12
#endif

#ifndef Oc
#define Oc 64
#endif

#ifndef Kh
#define Kh 3
#endif

#ifndef Kw
#define Kw 3
#endif

// TODO: include pad, stride, and dilation

#define Oh Ih - Kh + 1
#define Ow Iw - Kw + 1

float x[N][Ih][Iw][Ic];
float w[Oc][Kh][Kw][Ic];
float y[N][Oh][Ow][Oc];

__attribute__((noinline))
//template <N=32, Iw=...>
void conv2d(int* ret) {
  // loop over output
  for (int n = 0; n < N; n++) {
    for (int oh = 0; oh < Oh; oh++) {
      for (int ow = 0; ow < Ow; ow++) {
        for (int oc = 0; oc < Oc; oc++) {
          y[n][oh][ow][oc] = 0;
          // loop over filter
          for (int kh = 0; kh < Kh; kh++) {
            for (int kw = 0; kw < Kw; kw++) {
              for (int ic = 0; ic < Iw; ic++) {
                // TODO: include pad, stride, and dilation
                y[n][oh][ow][oc] += w[oc][kh][kw][ic] * x[n][oh - kh + 1][ow - kw + 1][ic];
              }
            }
          }
        }
      }
    }
  }
  *ret = y[N - 1][Oh - 1][Ow - 1][Oc - 1];
}

__attribute__((optnone)) int main(int argc, char* argv[]) {
  int dummy = 0;
  // TODO: initialize tensors
  BENCH("conv2d", conv2d(&dummy), 100, dummy);

  return 0;
}
