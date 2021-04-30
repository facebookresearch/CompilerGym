# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from threading import Lock

_executor_lock = Lock()
_executor = None


def get_thread_pool_executor() -> ThreadPoolExecutor:
    """Return a singleton :code:`ThreadPoolExecutor`.

    This executor is intended to be used for multithreaded parallelism. The
    maximum number of threads in the pool is equal to the number of cores on the
    machine. This is based on the assumption that CompilerGym workloads are
    typically CPU bound and not I/O bound, so the number of active threads
    should correspond to the number of available cores.

    :returns: A thread pool executor.
    """
    with _executor_lock:
        global _executor
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=cpu_count())
        return _executor
