# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module contains a reusable pool of service connections."""
import atexit
import logging
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Dict, List, Set, Tuple

from compiler_gym.service.connection import CompilerGymServiceConnection, ConnectionOpts

logger = logging.getLogger(__name__)

# We identify connections by the binary path and set of connection opts.
ServiceConnectionCacheKey = Tuple[Path, ConnectionOpts]


class ServiceConnectionPoolBase:
    """A class that provides the base interface for service connection pools."""

    def acquire(
        self, endpoint: Path, opts: ConnectionOpts
    ) -> CompilerGymServiceConnection:
        return CompilerGymServiceConnection(
            endpoint=endpoint, opts=opts, owning_service_pool=self
        )

    def release(self, service: CompilerGymServiceConnection) -> None:
        pass


class ServiceConnectionPool(ServiceConnectionPoolBase):
    """An object pool for compiler service connections.

    This class implements a thread-safe pool for compiler service connections.
    This enables compiler service connections to be reused, avoiding the
    expensive initialization of a new service.

    There is a global instance of this class, available via the static
    :meth:`ServiceConnectionPool.get()
    <compier_gym.service.ServiceConnectionPool.get>` method. To use the pool,
    acquire a reference to the global instance, and call the
    :meth:`ServiceConnectionPool.acquire()
    <compier_gym.service.ServiceConnectionPool.acquire>` method to construct and
    return service connections:

        >>> pool = ServiceConnectionPool.get()
        >>> with pool.acquire(Path("/path/to/service"), ConnectionOpts()) as service:
        ...    # Do something with the service.

    When a service is closed (by calling :meth:`service.close()
    <compiler_gym.service.CompilerGymServiceConnection.close>`), it is
    automatically released back to the pool so that a future request for the
    same type of service will reuse the connection.

    :ivar pool: A pool of service connections that are ready for use.

    :vartype pool: Dict[Tuple[Path, ConnectionOpts],
        List[CompilerGymServiceConnection]]

    :ivar allocated: The set of service connections that are currently in use.

    :vartype allocated: Set[CompilerGymServiceConnection]
    """

    def __init__(self):
        """"""
        self._lock = Lock()
        self.pool: Dict[
            ServiceConnectionCacheKey, List[CompilerGymServiceConnection]
        ] = defaultdict(list)
        self.allocated: Set[CompilerGymServiceConnection] = set()

        # Add a flag to indicate a closed connection pool because of
        # out-of-order execution of destructors and the atexit callback.
        self.closed = False

        atexit.register(self.close)

    def acquire(
        self, endpoint: Path, opts: ConnectionOpts
    ) -> CompilerGymServiceConnection:
        """Acquire a service connection from the pool.

        If an existing connection is available in the pool, it is returned.
        Otherwise, a new connection is created.
        """
        key: ServiceConnectionCacheKey = (endpoint, opts)
        with self._lock:
            if self.closed:
                # This should never happen.
                raise TypeError("ServiceConnectionPool is closed")

            if self.pool[key]:
                service = self.pool[key].pop().acquire()
                logger.debug(
                    "Reusing %s, %d environments remaining in pool",
                    service.connection.url,
                    len(self.pool[key]),
                )
            else:
                # No free service connections, construct a new one.
                service = CompilerGymServiceConnection(
                    endpoint=endpoint, opts=opts, owning_service_pool=self
                )
                logger.debug("Created %s", service.connection.url)

            self.allocated.add(service)

        return service

    def release(self, service: CompilerGymServiceConnection) -> None:
        """Release a service connection back to the pool.

        .. note::

            This method is called automatically by the :meth:`service.close()
            <compiler_gym.service.CompilerGymServiceConnection.close>` method of
            acquired service connections. You do not have to call this method
            yourself.
        """
        key: ServiceConnectionCacheKey = (service.endpoint, service.opts)
        with self._lock:
            # During shutdown, the shutdown routine for this
            # ServiceConnectionPool may be called before the destructor of
            # the managed CompilerGymServiceConnection objects.
            if self.closed:
                return

            if service not in self.allocated:
                logger.debug("Discarding service that does not belong to pool")
                return

            self.allocated.remove(service)

            # Only managed processes have a process attribute.
            if hasattr(service.connection, "process"):
                # A dead service cannot be reused, discard it.
                if service.closed or service.connection.process.poll() is not None:
                    logger.debug("Discarding service with dead process")
                    return
            # A service that has been shutdown cannot be reused, discard it.
            if not service.connection:
                logger.debug("Discarding service that has no connection")
                return

            self.pool[key].append(service)

        logger.debug("Released %s, pool size %d", service.connection.url, self.size)

    def __contains__(self, service: CompilerGymServiceConnection):
        """Check if a service connection is managed by the pool."""
        key: ServiceConnectionCacheKey = (service.endpoint, service.opts)
        return service in self.allocated or service in self.pool[key]

    @property
    def size(self):
        """Return the total number of connections in the pool."""
        return sum(len(x) for x in self.pool.values()) + len(self.allocated)

    def __len__(self):
        return self.size

    def close(self) -> None:
        """Close the pool, terminating all connections.

        Once closed, the pool cannot be used again. It is safe to call this
        method more than once.
        """
        with self._lock:
            if self.closed:
                return

            try:
                logger.debug(
                    "Closing the service connection pool with %d cached and %d live connections",
                    self.size,
                    len(self.allocated),
                )
            except ValueError:
                # As this method is invoked by the atexit callback, the logger
                # may already have closed its streams, in which case a
                # ValueError is raised.
                pass

            for connections in self.pool.values():
                for connection in connections:
                    connection.shutdown()
            self.pool = defaultdict(list)
            for connection in self.allocated:
                connection.shutdown()
            self.allocated = set()
            self.closed = True

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "ServiceConnectionPool":
        """Support for "with" statement."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Support for "with" statement."""
        self.close()
        return False

    @staticmethod
    def get() -> "ServiceConnectionPool":
        """Return the global instance of the service connection pool."""
        return _SERVICE_CONNECTION_POOL

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size})"


_SERVICE_CONNECTION_POOL = ServiceConnectionPool()
