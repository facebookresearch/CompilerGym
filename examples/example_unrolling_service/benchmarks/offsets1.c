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

int main(int argc, char* argv[]) {
  int dummy = 0;
  example1(&dummy);

  return 0;
}
