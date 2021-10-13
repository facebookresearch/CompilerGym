# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from typing import Dict, Optional

import numpy as np

from compiler_gym.service.proto import Benchmark

MAX_SIZE_IN_BYTES = 512 * 104 * 1024

logger = logging.getLogger(__name__)


class BenchmarkCache:
    """An in-memory cache of Benchmark messages.

    This object caches Benchmark messages by URI. Once the cache reaches a
    predetermined size, benchmarks are evicted randomly until the capacity is
    reduced to 50%.
    """

    def __init__(
        self,
        max_size_in_bytes: int = MAX_SIZE_IN_BYTES,
        rng: Optional[np.random.Generator] = None,
    ):
        self._max_size_in_bytes = max_size_in_bytes
        self.rng = rng or np.random.default_rng()

        self._benchmarks: Dict[str, Benchmark] = {}
        self._size_in_bytes = 0

    def __getitem__(self, uri: str) -> Benchmark:
        """Get a benchmark by URI. Raises KeyError."""
        item = self._benchmarks.get(uri)
        if item is None:
            raise KeyError(uri)
        return item

    def __contains__(self, uri: str):
        """Whether URI is in cache."""
        return uri in self._benchmarks

    def __setitem__(self, uri: str, benchmark: Benchmark):
        """Add benchmark to cache."""
        # Remove any existing value to keep the cache size consistent.
        if uri in self._benchmarks:
            self._size_in_bytes -= self._benchmarks[uri].ByteSize()
            del self._benchmarks[uri]

        size = benchmark.ByteSize()
        if self.size_in_bytes + size > self.max_size_in_bytes:
            if size > self.max_size_in_bytes:
                logger.warning(
                    "Adding new benchmark with size %d bytes exceeds total "
                    "target cache size of %d bytes",
                    size,
                    self.max_size_in_bytes,
                )
            else:
                logger.debug(
                    "Adding new benchmark with size %d bytes "
                    "exceeds maximum size %d bytes, %d items",
                    size,
                    self.max_size_in_bytes,
                    self.size,
                )
            self.evict_to_capacity()

        self._benchmarks[uri] = benchmark
        self._size_in_bytes += size

        logger.debug(
            "Cached benchmark %s. Cache size = %d bytes, %d items",
            uri,
            self.size_in_bytes,
            self.size,
        )

    def evict_to_capacity(self, target_size_in_bytes: Optional[int] = None) -> None:
        """Evict benchmarks randomly to reduce the capacity below 50%."""
        evicted = 0
        target_size_in_bytes = (
            self.max_size_in_bytes // 2
            if target_size_in_bytes is None
            else target_size_in_bytes
        )

        while self.size and self.size_in_bytes > target_size_in_bytes:
            evicted += 1
            key = self.rng.choice(list(self._benchmarks.keys()))
            self._size_in_bytes -= self._benchmarks[key].ByteSize()
            del self._benchmarks[key]

        if evicted:
            logger.info(
                "Evicted %d benchmarks from cache. "
                "Benchmark cache size now %d bytes, %d items",
                evicted,
                self.size_in_bytes,
                self.size,
            )

    @property
    def size(self) -> int:
        """The number of items in the cache."""
        return len(self._benchmarks)

    @property
    def size_in_bytes(self) -> int:
        """The combined size of the elements in the cache, excluding the
        cache overhead.
        """
        return self._size_in_bytes

    @property
    def max_size_in_bytes(self) -> int:
        """The maximum size of the cache."""
        return self._max_size_in_bytes

    @max_size_in_bytes.setter
    def max_size_in_bytes(self, value: int) -> None:
        """Set a new maximum cache size."""
        self._max_size_in_bytes = value
        self.evict_to_capacity(target_size_in_bytes=value)
