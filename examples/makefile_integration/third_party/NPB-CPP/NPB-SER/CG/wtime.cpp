#include "wtime.hpp"

#include <sys/time.h>

void wtime(double* t) {
  static int sec = -1;
  struct timeval tv;
  gettimeofday(&tv, 0);
  if (sec < 0)
    sec = tv.tv_sec;
  *t = (tv.tv_sec - sec) + 1.0e-6 * tv.tv_usec;
}
