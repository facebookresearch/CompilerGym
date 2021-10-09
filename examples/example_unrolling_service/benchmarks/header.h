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

#define ALIGNED16 __attribute__((aligned(16)))

typedef int aint __attribute__((__aligned__(16)));

// Warmup and then measure.
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
    if (print_times)                             \
      printf("%ld", mtime);                      \
  }
#define Mi 1048576        // 1<<20
#define print_times true  // argc > 1;
void init_memory(void* start, void* end);
void init_memory_float(float* start, float* end);
unsigned digest_memory(void* start, void* end);
void atimer(const char* title, bool print);
