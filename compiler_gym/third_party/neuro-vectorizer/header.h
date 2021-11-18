/*
Copyright (c) 2019, Ameer Haj Ali (UC Berkeley), and Intel Corporation
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include <stdbool.h>
#include <stdio.h>
#include <sys/time.h>

/**
 * Warmup and then measure.
 *
 * Adapted from Neurovectorizer's implementation:
 * https://github.com/intel/neuro-vectorizer/blob/d1b068998c08865c59f1586845bb947229f70a51/training_data/header.h
 *
 * Which was in turn adapted from LLVM:
 * https://github.com/llvm/llvm-test-suite/blob/7eca159e29ca4308256ef6e35560a2d884ac6b01/SingleSource/UnitTests/Vectorizer/gcc-loops.cpp#L330-L336
 */
#define BENCH(NAME, RUN_LINE, ITER, DIGEST_LINE) \
  {                                              \
    struct timeval Start, End;                   \
    RUN_LINE;                                    \
    gettimeofday(&Start, 0);                     \
    for (int i = 0; i < (ITER); ++i) RUN_LINE;   \
    gettimeofday(&End, 0);                       \
    unsigned r = DIGEST_LINE;                    \
    long mtime, s, us;                           \
    s = End.tv_sec - Start.tv_sec;               \
    us = End.tv_usec - Start.tv_usec;            \
    mtime = (s * 1000 + us / 1000.0) + 0.5;      \
    printf("%ld", mtime);                        \
  }
