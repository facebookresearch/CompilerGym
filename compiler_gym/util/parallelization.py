# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Utilities for parallelization / threading / concurrency."""
from itertools import tee
from threading import Lock
from typing import Any, Iterable


class _ThreadSafeTee:
    """An extension of :code:`itertools.tee()` that uses a lock to ensure
    exclusive access to the iterator.
    """

    def __init__(self, tee_obj, lock):
        self.tee_obj = tee_obj
        self.lock = lock

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.tee_obj)

    def __copy__(self):
        return _ThreadSafeTee(self.tee_obj.__copy__(), self.lock)


def thread_safe_tee(iterable: Iterable[Any], n: int = 2):
    """An extension of :code:`itertools.tee()` that yields thread-safe iterators."""
    lock = Lock()
    return tuple(_ThreadSafeTee(tee_obj, lock) for tee_obj in tee(iterable, n))
