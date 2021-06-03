# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from time import time
from typing import Callable, Optional

from absl.logging import skip_log_prefix


def humanize_duration(seconds: float) -> str:
    """Format a time for humans."""
    value = abs(seconds)
    sign = "-" if seconds < 0 else ""
    if value < 1e-6:
        return f"{sign}{value*1e9:.1f}ns"
    elif value < 1e-3:
        return f"{sign}{value*1e6:.1f}us"
    if value < 1:
        return f"{sign}{value*1e3:.1f}ms"
    elif value < 60:
        return f"{sign}{value:.3f}s"
    else:
        return f"{sign}{value:.1f}s"


def humanize_duration_hms(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


class Timer:
    """A very simple scoped timer.

    Example:

        >>> with Timer() as timer:
                time.sleep(10)
            print(f"That took {timer}")
        That took 10.0s

    If you're feeling even more terse:

        >>> with Timer("Did stuff"):
                # do stuff ...
        Did stuff in 5.6ms

    You can control where the print out should be logged to:

        >>> with Timer("Did stuff", logging.getLogger().info)
                # do stuff ...
        [log] Did stuff in 11us
    """

    def __init__(
        self, label: Optional[str] = None, print_fn: Callable[[str], None] = print
    ):
        self._start_time = None
        self._elapsed = None
        self.label = label
        self.print_fn = print_fn

    def reset(self) -> "Timer":
        self._start_time = time()
        return self

    def __enter__(self) -> "Timer":
        return self.reset()

    @property
    def time(self) -> float:
        if self._elapsed:
            return self._elapsed
        elif self._start_time:
            return time() - self._start_time
        else:
            return 0

    @skip_log_prefix
    def __exit__(self, *args):
        self._elapsed = time() - self._start_time
        if self.label:
            self.print_fn(f"{self.label} in {self}")

    def __str__(self):
        return humanize_duration(self.time)
